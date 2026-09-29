"""AI Hub 한국어 SNS 어댑터 테스트.

실제 원본과 같은 구조의 작은 JSON을 만들어 변환 규칙을 확인한다.
"""

import json
import unicodedata
from pathlib import Path

import pytest

from evidence_trace.ingest.aihub_adapter import (
    AdapterStats,
    classify_kind,
    iter_records,
    main,
)
from evidence_trace.ingest.records import EvidenceRecord, MessageKind

DIALOGUE = {
    "header": {
        "dialogueInfo": {"dialogueID": "d1", "topic": "상거래(쇼핑)"},
        "participantsInfo": [{"participantID": "P01"}, {"participantID": "P02"}],
    },
    "body": [
        {"utteranceID": "U1", "turnID": "T1", "participantID": "P01",
         "date": "2017-11-11", "time": "14:18:00", "utterance": "#@이름# 입금했어?"},
        {"utteranceID": "U2", "turnID": "T2", "participantID": "P02",
         "date": "2017-11-11", "time": "14:23:00", "utterance": "#@시스템#송금#"},
        {"utteranceID": "U3", "turnID": "T2", "participantID": "P02",
         "date": "잘못된날짜", "time": "14:23:00", "utterance": "깨진 행"},
        {"utteranceID": "U4", "turnID": "T3", "participantID": "P01",
         "date": "2017-11-11", "time": "14:25", "utterance": "#@이모티콘#하하#"},
    ],
}


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """원본과 같은 구조의 JSON 파일 두 개가 든 데이터셋 폴더를 만든다.

    파일명 하나는 macOS처럼 NFD로 만들어 정규화 처리를 함께 확인한다.

    Args:
        tmp_path: pytest가 제공하는 임시 폴더.

    Returns:
        데이터셋 루트 폴더 경로.
    """
    root = tmp_path / "valid"
    folder = root / "[라벨]한국어SNS_valid"
    folder.mkdir(parents=True)
    second = json.loads(json.dumps(DIALOGUE))
    second["header"]["dialogueInfo"]["dialogueID"] = "d2"

    for name, dialogues in (("상거래(쇼핑).json", [DIALOGUE]), ("시사교육.json", [second])):
        nfd_name = unicodedata.normalize("NFD", name)
        (folder / nfd_name).write_text(
            json.dumps({"numberOfItems": 1, "data": dialogues}, ensure_ascii=False),
            encoding="utf-8",
        )
    return root


def test_content_is_preserved_verbatim(dataset: Path) -> None:
    """비식별화 표시를 포함한 본문이 한 글자도 바뀌지 않는다."""
    records = list(iter_records(dataset))
    assert records[0].content == "#@이름# 입금했어?"


def test_malformed_utterance_is_skipped_and_counted(dataset: Path) -> None:
    """형식이 잘못된 발화는 레코드가 되지 않고 사유가 통계에 남는다."""
    stats = AdapterStats()
    records = list(iter_records(dataset, stats=stats))
    assert stats.dialogues == 2
    assert stats.records == len(records) == 6
    assert stats.skipped["ValueError"] == 2
    assert all(not r.record_id.endswith(":U3") for r in records)


def test_identifiers_are_globally_unique(dataset: Path) -> None:
    """대화가 달라도 같은 P01이 같은 사람으로 합쳐지지 않는다."""
    records = list(iter_records(dataset))
    senders = {r.sender for r in records if r.source_ref["utterance_id"] == "U1"}
    assert senders == {"aihub_sns:d1:P01", "aihub_sns:d2:P01"}
    assert len({r.record_id for r in records}) == len(records)


def test_recipients_exclude_sender(dataset: Path) -> None:
    """수신자 목록에는 대화의 다른 참여자만 들어간다."""
    first = next(iter_records(dataset))
    assert first.recipients == ("aihub_sns:d1:P02",)


def test_timestamp_is_kst_and_accepts_short_time(dataset: Path) -> None:
    """시각은 KST로 해석되고 초가 없는 형식도 받아들인다."""
    records = list(iter_records(dataset))
    assert records[0].timestamp.isoformat() == "2017-11-11T14:18:00+09:00"
    assert records[2].timestamp.isoformat() == "2017-11-11T14:25:00+09:00"


def test_source_ref_points_back_to_original(dataset: Path) -> None:
    """source_ref만으로 원본 파일과 발화를 찾아갈 수 있다(파일명은 NFC)."""
    ref = next(iter_records(dataset)).source_ref
    assert ref == {
        "dataset": "aihub_sns",
        "file": "[라벨]한국어SNS_valid/상거래(쇼핑).json",
        "dialogue_id": "d1",
        "utterance_id": "U1",
    }
    assert unicodedata.is_normalized("NFC", ref["file"])


def test_max_dialogues_limits_output(dataset: Path) -> None:
    """max_dialogues만큼만 대화를 처리한다."""
    records = list(iter_records(dataset, max_dialogues=1))
    assert {r.thread_id for r in records} == {"aihub_sns:d1"}


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("#@시스템#사진#", MessageKind.PHOTO),
        ("#@시스템#삭제#", MessageKind.DELETED),
        ("#@시스템#송금#", MessageKind.TRANSFER),
        ("#@시스템#기타#", MessageKind.OTHER_SYSTEM),
        ("#@이모티콘#", MessageKind.EMOTICON),
        ("#@이모티콘#흑흑# #@이모티콘#", MessageKind.EMOTICON),
        ("ㅋㅋ #@이모티콘#", MessageKind.TEXT),
        ("#@금융#로 보내줘", MessageKind.TEXT),
    ],
)
def test_classify_kind(utterance: str, expected: MessageKind) -> None:
    """시스템·이모티콘 표시에 따라 메시지 종류가 정해진다."""
    assert classify_kind(utterance) is expected


def test_cli_writes_verifiable_jsonl(dataset: Path, tmp_path: Path) -> None:
    """CLI가 만든 JSONL의 모든 줄이 무결성 검증을 통과한다."""
    out = tmp_path / "out.jsonl"
    assert main([str(dataset), str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    for line in lines:
        EvidenceRecord.from_dict(json.loads(line))


def test_missing_root_fails_cleanly(tmp_path: Path) -> None:
    """없는 폴더를 주면 예외 대신 종료 코드 1을 돌려준다."""
    assert main([str(tmp_path / "nope"), str(tmp_path / "out.jsonl")]) == 1
