"""AI Hub '한국어 SNS' 라벨링 데이터를 공통 증거 레코드로 변환한다.

원본 JSON은 파일 하나가 최대 수 GB라서 통째로 읽지 않고 ``ijson``으로 대화를
하나씩 스트리밍한다. 원본 구조는 다음과 같다::

    {
      "numberOfItems": 11247,
      "data": [
        {
          "header": {
            "dialogueInfo": {"dialogueID": "...", ...},
            "participantsInfo": [{"participantID": "P01", ...}, ...]
          },
          "body": [
            {"utteranceID": "U1", "participantID": "P01",
             "date": "2017-11-11", "time": "14:18:00", "utterance": "아 헐"},
            ...
          ]
        },
        ...
      ]
    }

변환 원칙:

* 본문(``utterance``)은 한 글자도 바꾸지 않는다. 비식별화 표시
  (``#@이름#``, ``#@시스템#송금#`` 등)도 그대로 둔다.
* 원본에는 기기 소유자 개념이 없으므로 ``direction``은 비워 둔다. 소유자
  지정과 날짜 이동은 가상 기기를 조립하는 다음 단계의 일이다.
* 원본 시각에는 시간대가 없어 한국 표준시(KST, UTC+9)로 가정한다.
* 참여자 ID(P01 등)는 대화마다 다른 사람이므로 대화 ID를 붙여 전역에서
  유일하게 만든다.
* 형식이 잘못된 발화는 버리지 않고 건너뛴 사유를 통계로 남긴다.

Example:
    프로젝트 루트에서 valid 데이터 일부를 JSONL로 변환한다::

        $ python -m evidence_trace.ingest.aihub_adapter \\
            data/raw/aihub_sns/extracted/valid \\
            data/processed/aihub_valid_sample.jsonl --max-dialogues 1000

    주제마다 200개씩 층화 표본을 만든다::

        $ python -m evidence_trace.ingest.aihub_adapter \\
            data/raw/aihub_sns/extracted/valid \\
            data/processed/aihub_valid_stratified.jsonl --max-per-file 200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import ijson

from evidence_trace.ingest.records import EvidenceRecord, MessageKind, SourceType

DATASET_NAME = "aihub_sns"
"""str: 레코드의 ``app``과 ``source_ref['dataset']``에 쓰는 데이터셋 이름."""

KST = timezone(timedelta(hours=9))
"""timezone: 원본 시각에 적용하는 한국 표준시."""

SYSTEM_MARKER_KINDS: dict[str, MessageKind] = {
    "#@시스템#사진#": MessageKind.PHOTO,
    "#@시스템#동영상#": MessageKind.VIDEO,
    "#@시스템#파일#": MessageKind.FILE,
    "#@시스템#송금#": MessageKind.TRANSFER,
    "#@시스템#삭제#": MessageKind.DELETED,
    "#@시스템#지도#": MessageKind.MAP,
    "#@시스템#검색#": MessageKind.SEARCH,
}
"""dict[str, MessageKind]: 시스템 표시와 메시지 종류의 대응표."""

_ANY_SYSTEM_MARKER = re.compile(r"#@시스템#[^#\s]+#")
_EMOTICON_ONLY = re.compile(r"^(?:\s*#@이모티콘#(?:[^#\s]+#)?)+\s*$")


@dataclass
class AdapterStats:
    """변환 과정의 집계 결과.

    Attributes:
        dialogues: 처리한 대화 수.
        records: 만들어진 레코드 수.
        skipped: 건너뛴 발화 수를 사유별로 센 카운터.
    """

    dialogues: int = 0
    records: int = 0
    skipped: Counter[str] = field(default_factory=Counter)


def classify_kind(utterance: str) -> MessageKind:
    """발화 본문을 보고 메시지 종류를 판별한다.

    시스템 표시가 있으면 그 종류를, 이모티콘 표시로만 이루어져 있으면
    이모티콘을, 그 밖에는 일반 텍스트로 본다. 대응표에 없는 시스템 표시
    (``#@시스템#기타#`` 등)는 기타 시스템 메시지로 분류한다.

    Args:
        utterance: 원본 발화 본문.

    Returns:
        판별된 메시지 종류.

    Example:
        >>> classify_kind("#@시스템#송금#")
        <MessageKind.TRANSFER: 'transfer'>
        >>> classify_kind("#@이모티콘#흑흑# #@이모티콘#")
        <MessageKind.EMOTICON: 'emoticon'>
        >>> classify_kind("#@이름# 입금했어")
        <MessageKind.TEXT: 'text'>
    """
    for marker, kind in SYSTEM_MARKER_KINDS.items():
        if marker in utterance:
            return kind
    if _ANY_SYSTEM_MARKER.search(utterance):
        return MessageKind.OTHER_SYSTEM
    if _EMOTICON_ONLY.match(utterance):
        return MessageKind.EMOTICON
    return MessageKind.TEXT


def parse_timestamp(date: str, time: str) -> datetime:
    """원본의 날짜·시각 문자열을 KST 시간대가 붙은 시각으로 바꾼다.

    Args:
        date: ``"YYYY-MM-DD"`` 형식 날짜.
        time: ``"HH:MM:SS"`` 또는 ``"HH:MM"`` 형식 시각.

    Returns:
        KST 시간대가 지정된 시각.

    Raises:
        ValueError: 날짜나 시각 형식이 올바르지 않은 경우.

    Example:
        >>> parse_timestamp("2017-11-11", "14:18:00").isoformat()
        '2017-11-11T14:18:00+09:00'
    """
    text = f"{date.strip()} {time.strip()}"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=KST)
        except ValueError:
            continue
    raise ValueError(f"시각 형식을 해석할 수 없습니다: {text!r}")


def _participant_key(dialogue_id: str, participant_id: str) -> str:
    """대화 안에서만 유일한 참여자 ID를 전역에서 유일한 식별자로 바꾼다.

    Args:
        dialogue_id: 원본 대화 ID.
        participant_id: 원본 참여자 ID (예: ``"P01"``).

    Returns:
        ``"aihub:<dialogue_id>:<participant_id>"`` 형식 문자열.
    """
    return f"{DATASET_NAME}:{dialogue_id}:{participant_id}"


def dialogue_to_records(
    dialogue: dict[str, Any],
    source_file: str,
    stats: AdapterStats | None = None,
) -> list[EvidenceRecord]:
    """원본 대화 하나를 증거 레코드 목록으로 바꾼다.

    Args:
        dialogue: 원본 ``data`` 배열의 원소 하나 (``header``와 ``body`` 포함).
        source_file: 데이터셋 루트 기준 원본 파일의 상대 경로. NFC로 정규화된
            값을 넘긴다.
        stats: 건너뛴 발화 수를 기록할 통계 객체. None이면 기록하지 않는다.

    Returns:
        발화 순서대로 정렬된 레코드 목록. 형식이 잘못된 발화는 빠진다.

    Raises:
        KeyError: 대화에 ``header.dialogueInfo.dialogueID``나 ``body``가 없는 경우.
    """
    header = dialogue["header"]
    dialogue_id = str(header["dialogueInfo"]["dialogueID"])
    thread_id = f"{DATASET_NAME}:{dialogue_id}"
    participants = [
        _participant_key(dialogue_id, str(p["participantID"]))
        for p in header.get("participantsInfo", [])
    ]

    records: list[EvidenceRecord] = []
    for utterance in dialogue["body"]:
        try:
            utterance_id = str(utterance["utteranceID"])
            sender = _participant_key(dialogue_id, str(utterance["participantID"]))
            timestamp = parse_timestamp(str(utterance["date"]), str(utterance["time"]))
            content = str(utterance["utterance"])
        except (KeyError, ValueError) as exc:
            if stats is not None:
                stats.skipped[type(exc).__name__] += 1
            continue

        records.append(
            EvidenceRecord(
                record_id=f"{thread_id}:{utterance_id}",
                source_type=SourceType.MESSENGER,
                app=DATASET_NAME,
                thread_id=thread_id,
                timestamp=timestamp,
                sender=sender,
                content=content,
                source_ref={
                    "dataset": DATASET_NAME,
                    "file": source_file,
                    "dialogue_id": dialogue_id,
                    "utterance_id": utterance_id,
                },
                kind=classify_kind(content),
                recipients=tuple(p for p in participants if p != sender),
            )
        )
    return records


def iter_dialogues(path: Path) -> Iterator[dict[str, Any]]:
    """원본 JSON 파일에서 대화를 하나씩 스트리밍으로 읽는다.

    Args:
        path: AI Hub 라벨링 JSON 파일 경로.

    Yields:
        ``data`` 배열의 원소(대화) 하나.

    Raises:
        ijson.JSONError: 파일이 올바른 JSON이 아닌 경우.
    """
    with path.open("rb") as handle:
        yield from ijson.items(handle, "data.item")


def iter_records(
    root: Path,
    max_dialogues: int | None = None,
    stats: AdapterStats | None = None,
    max_per_file: int | None = None,
) -> Iterator[EvidenceRecord]:
    """데이터셋 폴더 아래 모든 JSON 파일의 레코드를 차례로 생성한다.

    파일은 NFC로 정규화한 상대 경로 순으로 처리해 실행할 때마다 순서가
    같다. macOS의 NFD 파일명과 섞여도 결과가 바뀌지 않는다. AI Hub는
    주제별로 파일이 나뉘어 있으므로 ``max_per_file``을 주면 주제마다
    같은 수의 대화를 뽑는 층화 표본이 된다.

    Args:
        root: 압축을 푼 데이터셋 폴더 (예: ``.../extracted/valid``).
        max_dialogues: 전체에서 처리할 최대 대화 수. None이면 제한 없음.
        stats: 집계를 기록할 통계 객체. None이면 기록하지 않는다.
        max_per_file: 파일(주제)마다 처리할 최대 대화 수. None이면 제한 없음.

    Yields:
        변환된 증거 레코드.

    Raises:
        FileNotFoundError: ``root``가 폴더가 아니거나 JSON 파일이 하나도 없는 경우.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"데이터셋 폴더를 찾을 수 없습니다: {root}")

    files = sorted(
        (unicodedata.normalize("NFC", p.relative_to(root).as_posix()), p)
        for p in root.rglob("*.json")
    )
    if not files:
        raise FileNotFoundError(f"JSON 파일이 없습니다: {root}")

    seen = 0
    for rel_path, path in files:
        in_file = 0
        for dialogue in iter_dialogues(path):
            if max_dialogues is not None and seen >= max_dialogues:
                return
            if max_per_file is not None and in_file >= max_per_file:
                break
            seen += 1
            in_file += 1
            records = dialogue_to_records(dialogue, rel_path, stats)
            if stats is not None:
                stats.dialogues += 1
                stats.records += len(records)
            yield from records


def write_jsonl(records: Iterator[EvidenceRecord], out_path: Path) -> int:
    """레코드를 한 줄에 하나씩 JSONL 파일로 저장한다.

    Args:
        records: 저장할 레코드 이터레이터.
        out_path: 저장할 파일 경로. 상위 폴더가 없으면 만든다.

    Returns:
        저장한 레코드 수.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    """명령행 진입점. 데이터셋 폴더를 JSONL로 변환하고 통계를 출력한다.

    Args:
        argv: 명령행 인자 목록. None이면 ``sys.argv[1:]``을 쓴다.

    Returns:
        프로세스 종료 코드. 성공하면 0, 실패하면 1.
    """
    parser = argparse.ArgumentParser(description="AI Hub 한국어 SNS → 공통 증거 레코드 JSONL")
    parser.add_argument("root", type=Path, help="압축을 푼 데이터셋 폴더")
    parser.add_argument("out", type=Path, help="저장할 JSONL 경로")
    parser.add_argument("--max-dialogues", type=int, default=None, help="처리할 최대 대화 수")
    parser.add_argument(
        "--max-per-file", type=int, default=None, help="파일(주제)마다 처리할 최대 대화 수"
    )
    args = parser.parse_args(argv)

    stats = AdapterStats()
    try:
        records = iter_records(args.root, args.max_dialogues, stats, args.max_per_file)
        written = write_jsonl(records, args.out)
    except FileNotFoundError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    print(f"대화 {stats.dialogues:,d}개 → 레코드 {written:,d}건 저장: {args.out}")
    if stats.skipped:
        print(f"건너뛴 발화: {dict(stats.skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
