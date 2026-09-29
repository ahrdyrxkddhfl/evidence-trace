"""사건 시나리오로부터 합성 증거 대화를 생성한다.

역할 분담이 이 모듈의 핵심이다.

* **LLM**은 메시지 문장만 쓴다. 매번 AI Hub 실제 대화 몇 개를 예시로
  받아 말투를 따라 한다.
* **코드**는 시나리오대로 메시지를 배치하고 시각을 정하며, 시스템
  메시지(``#@시스템#송금#`` 등)를 직접 넣는다. 정답지(어느 레코드가
  어느 증거인가)도 코드가 배치한 위치로 기록한다. LLM의 주장은 정답에
  쓰이지 않는다.
* **후처리**는 LLM이 프롬프트만으로는 잘 지키지 못하는 규칙을 결정적으로
  맞춘다. 40자가 넘는 메시지를 문장부호 위치에서 쪼개 여러 메시지로 보내게
  하고(실제 사람이 끊어 보내는 방식), 이모지를 지운다(AI Hub가 원본의
  이모티콘을 가린 가공과 같은 방향).
* **검사기**는 실제 데이터 통계로 정한 기준(한 메시지 40자 이하, 이모지
  1% 수준, 시스템 표시 금지 등)으로 후처리 결과를 걸러 불합격이면 다시
  생성한다.

산출물(기본 위치 ``data/scenarios/<id>/generated/``):

* ``records.jsonl``: 합성 증거 레코드. 정답 정보는 들어 있지 않다.
* ``answers.json``: 증거 ID → 레코드 ID 대응표와 오답 후보(decoy) 목록.
* ``review.md``: 사람이 검수하기 위한 대화록.
* ``generation_log.jsonl``: 생성 시도마다의 합격 여부와 반려 사유.

Example:
    Ollama 서버가 켜져 있는 상태에서 프로젝트 루트에서 실행한다::

        $ python -m evidence_trace.ingest.synthetic \\
            data/scenarios/fraud_case_01/scenario.yaml \\
            --fewshot data/processed/aihub_valid_stratified.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

from evidence_trace.ingest.aihub_adapter import classify_kind
from evidence_trace.ingest.records import EvidenceRecord, SourceType

DATASET_NAME = "synthetic"
"""str: 합성 레코드의 ``app``과 ``source_ref['dataset']``에 쓰는 이름."""

MAX_MESSAGE_CHARS = 40
"""int: 한 메시지의 최대 글자 수. AI Hub 실제 메시지의 99% 지점(39자) 기준."""

MAX_EMOJI_MESSAGES_PER_BEAT = 1
"""int: 한 번의 생성에서 허용하는 이모지 포함 메시지 수. 실제 비율은 약 1%."""

SAY_MESSAGE_RANGE = (1, 3)
"""tuple[int, int]: 핵심 메시지(say)를 나눠 보낼 수 있는 메시지 수 범위."""

ALLOWED_TEXT_MARKERS = frozenset({
    "#@이름#", "#@금융#", "#@전번#", "#@주소#", "#@URL#", "#@이모티콘#",
    "#@번호#", "#@계정#", "#@소속#", "#@신원#", "#@기타#",
})
"""frozenset[str]: LLM이 본문에 쓸 수 있는 비식별화 표시.

AI Hub 원본에서 확인된 표시 중 시스템 표시를 뺀 것이다. 시스템 표시
(``#@시스템#송금#`` 등)는 코드만 ``say``/``event`` 장면으로 넣는다.
"""

_MARKER = re.compile(r"#@[^#\s]+#(?:[^#\s]+#)?")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_EMOJI_JOINERS = re.compile("[\uFE0F\u200D]")
_SPLIT_POINT = re.compile(r"(?<=[.!?~,])\s+")
_SPEAKER_LABELS = "ABCDEFGH"


class ScenarioError(ValueError):
    """시나리오 파일의 내용이 규칙에 맞지 않을 때 발생한다."""


class GenerationError(RuntimeError):
    """재시도 한도 안에 검사기를 통과하는 출력을 얻지 못했을 때 발생한다."""


# ---------------------------------------------------------------------------
# 시나리오 모델
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Person:
    """시나리오 등장인물.

    Attributes:
        id: 시나리오 안에서 쓰는 인물 식별자 (예: ``"victim_a"``).
        role: 역할 설명. LLM 프롬프트에 인물 설명으로 들어간다.
        contact_name: 휴대폰 연락처에 저장된 이름. 기기 소유자는 ``None``.
    """

    id: str
    role: str
    contact_name: str | None


@dataclass(frozen=True)
class Beat:
    """대화방 안의 한 장면.

    Attributes:
        index: 대화방 안에서의 순번(0부터).
        type: ``"say"``(정답이 되는 핵심 메시지), ``"chat"``(맥락 대화),
            ``"event"``(코드가 넣는 고정 문구, 증거 아님. 오답 후보 대화의
            송금 알림 등) 중 하나.
        at: 장면이 시작되는 시각.
        speakers: 이 장면에서 말하는 인물 ID들. say는 한 명이다.
        text: say의 ``must_convey``, chat의 ``intent``, event의 ``literal``.
        message_range: 생성할 메시지 수의 (최소, 최대).
        evidence_id: say의 증거 ID. chat은 ``None``.
        literal: 지정되면 LLM 없이 이 문자열을 그대로 메시지로 쓴다.
        markers: 반드시 본문에 들어가야 하는 비식별화 표시.
        tags: 증거 분류 태그.
    """

    index: int
    type: str
    at: datetime
    speakers: tuple[str, ...]
    text: str
    message_range: tuple[int, int]
    evidence_id: str | None = None
    literal: str | None = None
    markers: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Thread:
    """합성 대화방 하나.

    Attributes:
        id: 대화방 식별자 (예: ``"t_victim_a"``).
        participants: 참여 인물 ID들.
        style: 말투 지시문.
        decoy: 사건과 무관한 오답 후보 대화방이면 True.
        beats: 시각 순으로 정렬된 장면들.
    """

    id: str
    participants: tuple[str, ...]
    style: str
    decoy: bool
    beats: tuple[Beat, ...]


@dataclass(frozen=True)
class Scenario:
    """사건 시나리오 전체.

    Attributes:
        id: 시나리오 식별자 (예: ``"fraud_case_01"``).
        title: 사건 이름.
        split: ``dev`` / ``validation`` / ``test`` 중 하나.
        summary: 사건 개요. LLM 프롬프트의 배경 설명으로 쓴다.
        tz: 시나리오 시각에 적용하는 시간대.
        owner: 기기 소유자 인물 ID.
        persons: 인물 ID → 인물.
        threads: 합성 대화방들.
    """

    id: str
    title: str
    split: str
    summary: str
    tz: timezone
    owner: str
    persons: dict[str, Person]
    threads: tuple[Thread, ...]


def _parse_range(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    """``"6-8"``, ``7``, None 같은 메시지 수 표기를 (최소, 최대)로 바꾼다.

    Args:
        value: YAML에 적힌 값.
        default: 값이 없을 때 쓸 범위.

    Returns:
        (최소, 최대) 튜플.

    Raises:
        ScenarioError: 해석할 수 없거나 최소가 최대보다 큰 경우.

    Example:
        >>> _parse_range("6-8", (1, 1))
        (6, 8)
        >>> _parse_range(3, (1, 1))
        (3, 3)
    """
    if value is None:
        return default
    try:
        if isinstance(value, int):
            low = high = value
        else:
            parts = str(value).split("-")
            low, high = int(parts[0]), int(parts[-1])
    except ValueError as exc:
        raise ScenarioError(f"메시지 수를 해석할 수 없습니다: {value!r}") from exc
    if not 1 <= low <= high:
        raise ScenarioError(f"메시지 수 범위가 올바르지 않습니다: {value!r}")
    return low, high


def _parse_tz(value: str) -> timezone:
    """``"+09:00"`` 형식 문자열을 시간대로 바꾼다.

    Args:
        value: ``±HH:MM`` 형식 문자열.

    Returns:
        해당 오프셋의 시간대.

    Raises:
        ScenarioError: 형식이 올바르지 않은 경우.
    """
    match = re.fullmatch(r"([+-])(\d{2}):(\d{2})", str(value))
    if not match:
        raise ScenarioError(f"timezone 형식이 올바르지 않습니다: {value!r}")
    sign = 1 if match.group(1) == "+" else -1
    return timezone(sign * timedelta(hours=int(match.group(2)), minutes=int(match.group(3))))


def _parse_at(value: Any, tz: timezone) -> datetime:
    """장면 시각을 시간대가 붙은 datetime으로 바꾼다.

    Args:
        value: ``"2026-03-01T21:40"`` 형식 문자열 또는 YAML이 해석한 datetime.
        tz: 시간대 정보가 없을 때 붙일 시간대.

    Returns:
        시간대가 지정된 시각.

    Raises:
        ScenarioError: 해석할 수 없는 경우.
    """
    try:
        moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ScenarioError(f"시각을 해석할 수 없습니다: {value!r}") from exc
    return moment if moment.tzinfo else moment.replace(tzinfo=tz)


def load_scenario(path: Path) -> Scenario:
    """시나리오 YAML을 읽고 규칙을 검사해 :class:`Scenario`로 만든다.

    검사 규칙은 다음과 같다.

    * 모든 발화자는 해당 대화방의 참여자여야 한다.
    * 증거 ID는 시나리오 전체에서 유일해야 한다.
    * 대화방 안의 장면은 시각 순이어야 하고 사건 기간 안에 있어야 한다.
    * say와 event 장면은 발화자가 정확히 한 명이어야 한다.
    * event 장면은 ``literal``이 있어야 한다.
    * chat의 흐름 설명에 시스템 표시(``#@시스템#``)를 쓸 수 없다. LLM은
      시스템 표시를 쓰지 못하므로 모순된 지시가 되기 때문이다. 시스템
      메시지가 필요하면 event 장면으로 넣는다.

    기기 소유자는 ``owner`` 키로 지정한다. 없으면 ``contact_name``이
    비어 있는 유일한 인물을 소유자로 본다.

    Args:
        path: 시나리오 YAML 경로.

    Returns:
        검사를 통과한 시나리오.

    Raises:
        FileNotFoundError: 파일이 없는 경우.
        ScenarioError: 규칙을 어긴 경우. 메시지에 위반 위치가 담긴다.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tz = _parse_tz(data.get("timezone", "+09:00"))
    period = data["period"]
    start = _parse_at(f"{period['start']}T00:00", tz)
    end = _parse_at(f"{period['end']}T23:59", tz)

    persons = {
        p["id"]: Person(id=p["id"], role=p["role"], contact_name=p.get("contact_name"))
        for p in data["persons"]
    }
    owner = data.get("owner")
    if owner is None:
        owners = [p.id for p in persons.values() if p.contact_name is None]
        if len(owners) != 1:
            raise ScenarioError("owner가 없고 contact_name이 빈 인물도 정확히 한 명이 아닙니다")
        owner = owners[0]
    if owner not in persons:
        raise ScenarioError(f"owner가 인물 목록에 없습니다: {owner}")

    seen_evidence: set[str] = set()
    threads: list[Thread] = []
    for raw_thread in data["threads"]:
        thread_id = raw_thread["thread"]
        participants = tuple(raw_thread["participants"])
        unknown = [p for p in participants if p not in persons]
        if unknown:
            raise ScenarioError(f"{thread_id}: 인물 목록에 없는 참여자 {unknown}")

        beats: list[Beat] = []
        for index, raw in enumerate(raw_thread["beats"]):
            where = f"{thread_id} 장면 {index}"
            beat_type = raw["type"]
            at = _parse_at(raw["at"], tz)
            if not start <= at <= end:
                raise ScenarioError(f"{where}: 사건 기간 밖의 시각 {at.isoformat()}")
            if beats and at <= beats[-1].at:
                raise ScenarioError(f"{where}: 이전 장면보다 시각이 빠르거나 같습니다")

            if beat_type == "say":
                speakers = (raw["speaker"],)
                text = raw["must_convey"]
                evidence_id = raw["evidence_id"]
                if evidence_id in seen_evidence:
                    raise ScenarioError(f"{where}: 중복된 증거 ID {evidence_id}")
                seen_evidence.add(evidence_id)
                message_range = (1, 1) if raw.get("literal") else SAY_MESSAGE_RANGE
            elif beat_type == "chat":
                speakers = tuple(raw.get("speakers", participants))
                text = raw["intent"]
                if "#@시스템#" in str(text):
                    raise ScenarioError(
                        f"{where}: chat 흐름 설명에 시스템 표시가 있습니다. event 장면으로 분리하세요"
                    )
                evidence_id = None
                message_range = _parse_range(raw.get("messages"), (4, 6))
            elif beat_type == "event":
                if not raw.get("literal"):
                    raise ScenarioError(f"{where}: event 장면에는 literal이 필요합니다")
                speakers = (raw["speaker"],)
                text = raw["literal"]
                evidence_id = None
                message_range = (1, 1)
            else:
                raise ScenarioError(f"{where}: 알 수 없는 장면 종류 {beat_type!r}")

            outsiders = [s for s in speakers if s not in participants]
            if outsiders:
                raise ScenarioError(f"{where}: 대화방 참여자가 아닌 발화자 {outsiders}")

            beats.append(
                Beat(
                    index=index,
                    type=beat_type,
                    at=at,
                    speakers=speakers,
                    text=str(text).strip(),
                    message_range=message_range,
                    evidence_id=evidence_id,
                    literal=raw.get("literal"),
                    markers=tuple(raw.get("markers", ())),
                    tags=tuple(raw.get("tags", ())),
                )
            )

        threads.append(
            Thread(
                id=thread_id,
                participants=participants,
                style=raw_thread.get("style", ""),
                decoy=bool(raw_thread.get("decoy", False)),
                beats=tuple(beats),
            )
        )

    return Scenario(
        id=data["scenario_id"],
        title=data.get("title", ""),
        split=data.get("split", "dev"),
        summary=str(data.get("summary", "")).strip(),
        tz=tz,
        owner=owner,
        persons=persons,
        threads=tuple(threads),
    )


# ---------------------------------------------------------------------------
# LLM 연결
# ---------------------------------------------------------------------------


class ChatModel(Protocol):
    """생성기가 쓰는 LLM의 최소 인터페이스.

    테스트에서는 정해진 응답을 돌려주는 가짜 구현으로 바꿔 끼운다.
    """

    name: str

    def complete(self, system: str, user: str, seed: int) -> str:
        """프롬프트에 대한 응답 문자열을 돌려준다.

        Args:
            system: 시스템 프롬프트.
            user: 사용자 프롬프트.
            seed: 재현성을 위한 난수 시드.

        Returns:
            모델 응답 본문.
        """
        ...


@dataclass
class OllamaChatModel:
    """로컬 Ollama 서버의 ``/api/chat``을 호출하는 모델.

    응답을 JSON 스키마로 강제해 파싱 실패를 줄인다. 표준 라이브러리만
    쓰므로 추가 의존성이 없다.

    Attributes:
        name: Ollama 모델 이름 (예: ``"exaone3.5:7.8b"``).
        host: Ollama 서버 주소.
        temperature: 샘플링 온도.
        timeout: 요청 제한 시간(초).
    """

    name: str = "exaone3.5:7.8b"
    host: str = "http://localhost:11434"
    temperature: float = 0.9
    timeout: float = 300.0

    def complete(self, system: str, user: str, seed: int) -> str:
        """Ollama에 요청을 보내고 응답 본문을 돌려준다.

        Args:
            system: 시스템 프롬프트.
            user: 사용자 프롬프트.
            seed: 난수 시드.

        Returns:
            모델이 생성한 JSON 문자열.

        Raises:
            GenerationError: 서버에 연결할 수 없거나 응답 형식이 잘못된 경우.
        """
        payload = {
            "model": self.name,
            "stream": False,
            "format": OUTPUT_SCHEMA,
            "options": {"temperature": self.temperature, "seed": seed},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise GenerationError(
                f"Ollama 서버에 연결할 수 없습니다({self.host}). "
                "`brew services start ollama`로 서버를 켰는지 확인하세요."
            ) from exc
        try:
            return body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise GenerationError(f"예상하지 못한 Ollama 응답: {body}") from exc


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "text"],
            },
        }
    },
    "required": ["messages"],
}
"""dict: LLM 출력에 강제하는 JSON 스키마."""


# ---------------------------------------------------------------------------
# 예시(few-shot)와 프롬프트
# ---------------------------------------------------------------------------


@dataclass
class FewShotPool:
    """말투 예시로 쓸 실제 대화 묶음.

    Attributes:
        dialogues: 대화별 메시지 목록. 각 메시지는 (발화자 라벨, 본문)이다.
        lines: 예시에 쓰인 모든 본문. LLM이 예시를 그대로 베꼈는지 검사할 때 쓴다.
    """

    dialogues: list[list[tuple[str, str]]]
    lines: set[str] = field(default_factory=set)

    @classmethod
    def from_jsonl(cls, path: Path, max_messages: int = 10) -> FewShotPool:
        """공통 레코드 JSONL에서 대화 단위 예시를 만든다.

        Args:
            path: :mod:`aihub_adapter`가 만든 JSONL 경로.
            max_messages: 대화 하나에서 예시로 쓸 최대 메시지 수.

        Returns:
            대화가 하나 이상 담긴 예시 묶음.

        Raises:
            FileNotFoundError: 파일이 없는 경우.
            ValueError: 대화가 하나도 없는 경우.
        """
        threads: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                threads[record["thread_id"]].append(record)
        if not threads:
            raise ValueError(f"예시로 쓸 대화가 없습니다: {path}")

        dialogues: list[list[tuple[str, str]]] = []
        lines: set[str] = set()
        for records in threads.values():
            labels: dict[str, str] = {}
            dialogue = []
            for record in records[:max_messages]:
                label = labels.setdefault(record["sender"], _SPEAKER_LABELS[len(labels) % 8])
                dialogue.append((label, record["content"]))
                lines.add(record["content"].strip())
            dialogues.append(dialogue)
        return cls(dialogues=dialogues, lines=lines)

    def sample(self, rng: random.Random, k: int) -> list[list[tuple[str, str]]]:
        """예시 대화 k개를 무작위로 고른다.

        Args:
            rng: 재현 가능한 난수 생성기.
            k: 고를 대화 수. 가진 대화보다 많으면 전부 돌려준다.

        Returns:
            고른 대화 목록.
        """
        return rng.sample(self.dialogues, min(k, len(self.dialogues)))


SYSTEM_PROMPT = """\
너는 한국인의 실제 메신저 대화를 재현하는 작가다. 아래 규칙을 반드시 지킨다.

1. 예시 대화의 말투를 따라 한다. 한 메시지는 짧게(대부분 20자 이하, 최대 40자) 쓰고,
   한 사람이 여러 메시지로 끊어 보내는 경우가 많다. 오타, 줄임말, ㅋㅋ, ㅠㅠ를 자연스럽게 섞는다.
2. 사람 이름은 #@이름#, 계좌번호는 #@금융#, 전화번호는 #@전번#, 주소는 #@주소#, 링크는 #@URL#,
   송장번호 같은 그 밖의 번호는 #@번호#로 쓴다.
   이모티콘은 이모지 대신 #@이모티콘#으로 쓴다.
3. '#@시스템#'으로 시작하는 표시는 절대 쓰지 않는다.
4. 매 메시지마다 상대 이름을 부르지 않는다.
5. 출력은 {"messages": [{"speaker": "A", "text": "..."}]} 형식의 JSON만 쓴다.\
"""
"""str: 모든 생성 요청에 공통으로 쓰는 시스템 프롬프트."""


def _format_dialogue(dialogue: Iterable[tuple[str, str]]) -> str:
    """(라벨, 본문) 목록을 ``A: 본문`` 줄들로 바꾼다.

    Args:
        dialogue: (발화자 라벨, 본문) 목록.

    Returns:
        줄바꿈으로 이은 대화록.
    """
    return "\n".join(f"{label}: {text}" for label, text in dialogue)


def build_user_prompt(
    scenario: Scenario,
    thread: Thread,
    beat: Beat,
    history: list[tuple[str, str]],
    examples: list[list[tuple[str, str]]],
    labels: dict[str, str],
) -> str:
    """장면 하나를 생성하기 위한 사용자 프롬프트를 만든다.

    Args:
        scenario: 사건 시나리오. 개요를 배경으로 쓴다.
        thread: 장면이 속한 대화방.
        beat: 생성할 장면.
        history: 이 대화방에서 지금까지 생성된 (라벨, 본문) 목록.
        examples: 말투 예시로 보여줄 실제 대화들.
        labels: 인물 ID → 발화자 라벨(A, B, ...).

    Returns:
        LLM에 보낼 사용자 프롬프트.
    """
    people = "\n".join(
        f"- {labels[pid]}: {scenario.persons[pid].role}" for pid in thread.participants
    )
    shown = "\n\n".join(f"[예시 {i + 1}]\n{_format_dialogue(d)}" for i, d in enumerate(examples))
    recent = _format_dialogue(history[-12:]) if history else "(대화 시작)"
    low, high = beat.message_range
    count = f"{low}개" if low == high else f"{low}~{high}개"
    speakers = ", ".join(labels[s] for s in beat.speakers)

    if beat.type == "say":
        task = (
            f"{speakers}가 다음 내용을 전달하는 메시지를 {count} 써라. "
            f"여러 개로 끊어 보내도 된다.\n전달할 내용: {beat.text}"
        )
        if beat.markers:
            task += f"\n반드시 포함할 표시: {', '.join(beat.markers)}"
    else:
        task = f"다음 흐름의 대화를 메시지 {count}로 써라. 말하는 사람: {speakers}\n흐름: {beat.text}"

    return (
        f"[실제 대화 예시 — 말투만 참고하고 내용은 베끼지 마라]\n{shown}\n\n"
        f"[사건 배경]\n{scenario.summary}\n\n"
        f"[등장인물]\n{people}\n\n"
        f"[말투]\n{thread.style}\n\n"
        f"[지금까지의 대화]\n{recent}\n\n"
        f"[할 일]\n{task}\n\n"
        "[주의] 한 메시지는 20자 안팎으로 짧게 쓰고 40자를 넘기지 마라. "
        "긴 말은 여러 메시지로 끊어라. 이모지는 쓰지 마라."
    )


# ---------------------------------------------------------------------------
# 후처리
# ---------------------------------------------------------------------------


def split_long_text(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """긴 메시지를 문장부호 위치에서 ``limit`` 이하 조각들로 나눈다.

    문장부호(``. ! ? ~ ,``) 뒤 공백을 경계로 자른 뒤, 앞에서부터 ``limit``을
    넘지 않는 만큼 이어 붙인다. 경계가 없어 ``limit``을 넘는 조각은 그대로
    두며, 검사기가 반려한다.

    Args:
        text: 원래 메시지.
        limit: 한 조각의 최대 글자 수.

    Returns:
        조각 목록. ``text``가 ``limit`` 이하이면 ``[text]``.

    Example:
        >>> split_long_text("안녕하세요! 판매글 보고 연락드렸어요. 아직 판매 중인가요?", 20)
        ['안녕하세요!', '판매글 보고 연락드렸어요.', '아직 판매 중인가요?']
    """
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    for part in _SPLIT_POINT.split(text):
        if pieces and len(pieces[-1]) + 1 + len(part) <= limit:
            pieces[-1] = f"{pieces[-1]} {part}"
        else:
            pieces.append(part)
    return pieces


def normalize_messages(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    """LLM 출력을 실제 메신저 말투에 맞게 결정적으로 정리한다.

    이모지와 이모지 결합 문자를 지우고, 공백을 정리한 뒤, 40자를 넘는
    메시지는 :func:`split_long_text`로 쪼개 같은 발화자의 연속 메시지로 만든다.
    이모지를 지운 뒤 비어 버린 메시지는 버린다.

    Args:
        messages: ``{"speaker": 라벨, "text": 본문}`` 목록.

    Returns:
        정리된 메시지 목록과 ``{"split": 쪼갠 메시지 수, "emoji_removed": 이모지를
        지운 메시지 수}`` 통계.

    Example:
        >>> out, stats = normalize_messages([{"speaker": "A", "text": "좋아요😊"}])
        >>> out, stats
        ([{'speaker': 'A', 'text': '좋아요'}], {'split': 0, 'emoji_removed': 1})
    """
    stats = {"split": 0, "emoji_removed": 0}
    result: list[dict[str, str]] = []
    for message in messages:
        text = str(message.get("text", ""))
        cleaned = _EMOJI_JOINERS.sub("", _EMOJI.sub("", text))
        if cleaned != text:
            stats["emoji_removed"] += 1
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue
        pieces = split_long_text(cleaned)
        if len(pieces) > 1:
            stats["split"] += 1
        result.extend({"speaker": message.get("speaker", ""), "text": piece} for piece in pieces)
    return result, stats


# ---------------------------------------------------------------------------
# 검사기
# ---------------------------------------------------------------------------


def validate_messages(
    messages: list[dict[str, str]],
    beat: Beat,
    allowed_labels: set[str],
    fewshot_lines: set[str],
    check_count: bool = True,
) -> list[str]:
    """메시지들이 규칙을 지켰는지 검사한다.

    Args:
        messages: ``{"speaker": 라벨, "text": 본문}`` 목록.
        beat: 생성 대상 장면. 메시지 수와 필수 표시를 확인한다.
        allowed_labels: 이 장면에서 말할 수 있는 발화자 라벨.
        fewshot_lines: 예시로 보여준 실제 대화 본문. 그대로 베낀 문장을 막는다.
        check_count: True이면 메시지 수가 장면의 범위 안인지 확인한다.
            후처리로 쪼갠 뒤에는 개수가 늘어나므로, 개수 검사는 LLM 원본에
            대해서만 하고 후처리 결과에는 False로 부른다.

    Returns:
        위반 사유 목록. 비어 있으면 합격이다.

    Example:
        >>> beat = Beat(0, "chat", datetime(2026, 3, 1, tzinfo=timezone.utc),
        ...             ("a",), "인사", (1, 2))
        >>> validate_messages([{"speaker": "A", "text": "송금함 #@시스템#송금#"}],
        ...                   beat, {"A"}, set())
        ['메시지 1: 시스템 표시 사용 #@시스템#송금#']
    """
    problems: list[str] = []
    low, high = beat.message_range
    if check_count and not low <= len(messages) <= high:
        problems.append(f"메시지 수 {len(messages)}개 (허용 {low}~{high})")

    emoji_messages = 0
    for number, message in enumerate(messages, start=1):
        text = str(message.get("text", "")).strip()
        where = f"메시지 {number}"
        if not text:
            problems.append(f"{where}: 빈 메시지")
            continue
        if message.get("speaker") not in allowed_labels:
            problems.append(f"{where}: 허용되지 않은 발화자 {message.get('speaker')!r}")
        if len(text) > MAX_MESSAGE_CHARS:
            problems.append(f"{where}: {len(text)}자 (최대 {MAX_MESSAGE_CHARS}자)")
        if _EMOJI.search(text):
            emoji_messages += 1
        for marker in _MARKER.findall(text):
            if marker.startswith("#@시스템#"):
                problems.append(f"{where}: 시스템 표시 사용 {marker}")
            elif marker not in ALLOWED_TEXT_MARKERS and not marker.startswith("#@이모티콘#"):
                problems.append(f"{where}: 알 수 없는 표시 {marker}")
        if len(text) >= 6 and text in fewshot_lines:
            problems.append(f"{where}: 예시 문장을 그대로 베낌")

    if emoji_messages > MAX_EMOJI_MESSAGES_PER_BEAT:
        problems.append(f"이모지 포함 메시지 {emoji_messages}개 (최대 {MAX_EMOJI_MESSAGES_PER_BEAT}개)")

    joined = " ".join(str(m.get("text", "")) for m in messages)
    for marker in beat.markers:
        if marker not in joined:
            problems.append(f"필수 표시 {marker} 누락")
    return problems


def _parse_output(raw: str) -> list[dict[str, str]]:
    """LLM 응답에서 메시지 목록을 꺼낸다.

    Args:
        raw: LLM이 돌려준 문자열.

    Returns:
        메시지 목록.

    Raises:
        ValueError: JSON이 아니거나 ``messages`` 목록이 없는 경우.
    """
    data = json.loads(raw)
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list):
        raise ValueError("messages 목록이 없습니다")
    return [m for m in messages if isinstance(m, dict)]


# ---------------------------------------------------------------------------
# 시각 배치
# ---------------------------------------------------------------------------


def schedule_times(
    start: datetime,
    count: int,
    deadline: datetime | None,
    rng: random.Random,
    max_gap_minutes: int = 3,
) -> list[datetime]:
    """장면의 메시지들에 분 단위 시각을 배정한다.

    첫 메시지는 장면 시작 시각에 두고, 나머지는 메시지당 최대
    ``max_gap_minutes``분 안에서 무작위로 흩어 놓는다. 다음 장면의
    시작 시각(``deadline``)보다 1분 이상 앞서도록 제한한다. 실제 데이터처럼
    같은 분에 여러 메시지가 올 수 있다.

    Args:
        start: 장면 시작 시각.
        count: 메시지 수.
        deadline: 다음 장면 시작 시각. 없으면 제한 없음.
        rng: 재현 가능한 난수 생성기.
        max_gap_minutes: 메시지당 평균적으로 쓸 수 있는 최대 간격(분).

    Returns:
        오름차순으로 정렬된 시각 목록.

    Raises:
        ScenarioError: 다음 장면까지 남은 시간이 1분 미만인 경우.
    """
    budget = (count - 1) * max_gap_minutes
    if deadline is not None:
        room = int((deadline - start).total_seconds() // 60) - 1
        if room < 0:
            raise ScenarioError(f"{start.isoformat()} 장면과 다음 장면 사이 시간이 부족합니다")
        budget = min(budget, room)
    offsets = sorted([0] + [rng.randint(0, budget) for _ in range(count - 1)])
    return [start + timedelta(minutes=m) for m in offsets]


# ---------------------------------------------------------------------------
# 생성
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """생성 결과 묶음.

    Attributes:
        records: 합성 증거 레코드 (시각 순).
        answers: 정답지. ``evidence``와 ``decoys`` 키를 가진다.
        review_lines: 검수용 대화록 줄들.
        log: 생성 시도 기록.
    """

    records: list[EvidenceRecord]
    answers: dict[str, Any]
    review_lines: list[str]
    log: list[dict[str, Any]]


def _beat_seed(base: int, scenario_id: str, thread_id: str, index: int, attempt: int) -> int:
    """장면과 시도 번호로부터 재현 가능한 시드를 만든다.

    Args:
        base: 기본 시드.
        scenario_id: 시나리오 ID.
        thread_id: 대화방 ID.
        index: 장면 순번.
        attempt: 시도 번호.

    Returns:
        0 이상 2**31 미만의 정수 시드.
    """
    key = f"{base}:{scenario_id}:{thread_id}:{index}:{attempt}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16) % (2**31)


def generate(
    scenario: Scenario,
    model: ChatModel,
    fewshot: FewShotPool,
    seed: int = 20260301,
    max_attempts: int = 4,
    examples_per_prompt: int = 4,
    progress: Callable[[str], None] | None = None,
) -> GenerationResult:
    """시나리오의 모든 합성 대화방을 생성한다.

    장면마다 LLM 원본의 메시지 수를 먼저 검사하고, :func:`normalize_messages`로
    후처리한 뒤 나머지 규칙을 검사한다.

    Args:
        scenario: 생성할 시나리오.
        model: 메시지 문장을 쓸 LLM.
        fewshot: 말투 예시 묶음.
        seed: 재현성을 위한 기본 시드.
        max_attempts: 장면 하나당 최대 생성 시도 횟수.
        examples_per_prompt: 프롬프트마다 보여줄 예시 대화 수.
        progress: 진행 상황 문자열을 받을 함수. None이면 출력하지 않는다.

    Returns:
        레코드, 정답지, 검수용 대화록, 시도 기록.

    Raises:
        GenerationError: 어떤 장면이 ``max_attempts`` 안에 검사를 통과하지 못한 경우.
            마지막 시도의 위반 사유가 메시지에 담긴다.
    """
    rng = random.Random(seed)
    report = progress or (lambda _message: None)
    total_beats = sum(len(t.beats) for t in scenario.threads)
    done = 0
    records: list[EvidenceRecord] = []
    evidence: dict[str, Any] = {}
    decoys: dict[str, list[str]] = {}
    review: list[str] = [f"# {scenario.title} ({scenario.id}) 검수용 대화록", ""]
    log: list[dict[str, Any]] = []

    for thread in scenario.threads:
        labels = {pid: _SPEAKER_LABELS[i] for i, pid in enumerate(thread.participants)}
        by_label = {v: k for k, v in labels.items()}
        thread_key = f"{DATASET_NAME}:{scenario.id}:{thread.id}"
        history: list[tuple[str, str]] = []
        thread_record_ids: list[str] = []
        names = ", ".join(scenario.persons[p].contact_name or "기기 소유자" for p in thread.participants)
        review += [f"## {thread.id}{' (오답 후보)' if thread.decoy else ''} — {names}", ""]

        for position, beat in enumerate(thread.beats):
            done += 1
            where = f"[{done}/{total_beats}] {thread.id} 장면 {beat.index}"
            following = thread.beats[position + 1].at if position + 1 < len(thread.beats) else None

            if beat.literal is not None:
                messages = [{"speaker": labels[beat.speakers[0]], "text": beat.literal}]
                log.append({"thread": thread.id, "beat": beat.index, "attempt": 0, "ok": True,
                            "problems": [], "literal": True})
                report(f"{where}: 고정 문구")
            else:
                allowed = {labels[s] for s in beat.speakers}
                problems: list[str] = []
                messages = []
                for attempt in range(1, max_attempts + 1):
                    prompt = build_user_prompt(
                        scenario, thread, beat, history,
                        fewshot.sample(rng, examples_per_prompt), labels,
                    )
                    raw = model.complete(
                        SYSTEM_PROMPT, prompt,
                        _beat_seed(seed, scenario.id, thread.id, beat.index, attempt),
                    )
                    fixes = {"split": 0, "emoji_removed": 0}
                    try:
                        raw_messages = _parse_output(raw)
                        low, high = beat.message_range
                        if low <= len(raw_messages) <= high:
                            messages, fixes = normalize_messages(raw_messages)
                            problems = validate_messages(
                                messages, beat, allowed, fewshot.lines, check_count=False
                            )
                        else:
                            problems = [f"메시지 수 {len(raw_messages)}개 (허용 {low}~{high})"]
                    except (ValueError, json.JSONDecodeError) as exc:
                        problems = [f"출력 파싱 실패: {exc}"]
                    log.append({"thread": thread.id, "beat": beat.index, "attempt": attempt,
                                "ok": not problems, "problems": problems, **fixes})
                    status = "합격" if not problems else f"반려({problems[0]})"
                    report(f"{where} 시도 {attempt}: {status}")
                    if not problems:
                        break
                else:
                    raise GenerationError(
                        f"{thread.id} 장면 {beat.index}: {max_attempts}회 시도 모두 불합격 — {problems}"
                    )

            times = schedule_times(beat.at, len(messages), following, rng)
            beat_ids: list[str] = []
            for moment, message in zip(times, messages):
                sender_id = by_label[message["speaker"]]
                text = str(message["text"]).strip()
                record = EvidenceRecord(
                    record_id=f"{thread_key}:{len(thread_record_ids):04d}",
                    source_type=SourceType.MESSENGER,
                    app=DATASET_NAME,
                    thread_id=thread_key,
                    timestamp=moment,
                    sender=f"{DATASET_NAME}:{scenario.id}:{sender_id}",
                    content=text,
                    source_ref={
                        "dataset": DATASET_NAME,
                        "scenario": scenario.id,
                        "thread": thread.id,
                        "beat": str(beat.index),
                    },
                    kind=classify_kind(text),
                    recipients=tuple(
                        f"{DATASET_NAME}:{scenario.id}:{p}" for p in thread.participants if p != sender_id
                    ),
                )
                records.append(record)
                thread_record_ids.append(record.record_id)
                beat_ids.append(record.record_id)
                history.append((message["speaker"], text))
                who = scenario.persons[sender_id].contact_name or "소유자"
                mark = f" **[{beat.evidence_id}]**" if beat.evidence_id else ""
                review.append(f"- `{moment.strftime('%m-%d %H:%M')}` {who}: {text}{mark}")

            if beat.evidence_id:
                evidence[beat.evidence_id] = {
                    "record_ids": beat_ids,
                    "thread": thread.id,
                    "must_convey": beat.text,
                    "tags": list(beat.tags),
                }
        review.append("")
        if thread.decoy:
            decoys[thread.id] = thread_record_ids

    records.sort(key=lambda r: (r.timestamp, r.record_id))
    answers = {
        "scenario_id": scenario.id,
        "split": scenario.split,
        "model": model.name,
        "seed": seed,
        "evidence": evidence,
        "decoys": decoys,
    }
    return GenerationResult(records=records, answers=answers, review_lines=review, log=log)


def write_outputs(result: GenerationResult, out_dir: Path) -> None:
    """생성 결과를 파일로 저장한다.

    Args:
        result: :func:`generate`의 결과.
        out_dir: 저장할 폴더. 없으면 만든다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in result.records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    (out_dir / "answers.json").write_text(
        json.dumps(result.answers, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "review.md").write_text("\n".join(result.review_lines) + "\n", encoding="utf-8")
    with (out_dir / "generation_log.jsonl").open("w", encoding="utf-8") as handle:
        for entry in result.log:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    """명령행 진입점. 시나리오를 읽어 합성 대화를 생성하고 저장한다.

    Args:
        argv: 명령행 인자 목록. None이면 ``sys.argv[1:]``을 쓴다.

    Returns:
        프로세스 종료 코드. 성공하면 0, 실패하면 1.
    """
    parser = argparse.ArgumentParser(description="사건 시나리오 → 합성 증거 대화")
    parser.add_argument("scenario", type=Path, help="scenario.yaml 경로")
    parser.add_argument("--fewshot", type=Path, required=True, help="말투 예시용 공통 레코드 JSONL")
    parser.add_argument("--model", default="exaone3.5:7.8b", help="Ollama 모델 이름")
    parser.add_argument("--seed", type=int, default=20260301, help="기본 시드")
    parser.add_argument("--out", type=Path, default=None, help="저장 폴더 (기본: 시나리오 폴더/generated)")
    args = parser.parse_args(argv)

    try:
        scenario = load_scenario(args.scenario)
        pool = FewShotPool.from_jsonl(args.fewshot)
        result = generate(
            scenario, OllamaChatModel(name=args.model), pool, seed=args.seed,
            progress=lambda message: print(message, flush=True),
        )
    except (ScenarioError, GenerationError, FileNotFoundError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    out_dir = args.out or args.scenario.parent / "generated"
    write_outputs(result, out_dir)
    attempts = [e for e in result.log if not e.get("literal")]
    rejected = sum(1 for e in attempts if not e["ok"])
    accepted = [e for e in attempts if e["ok"]]
    split = sum(e.get("split", 0) for e in accepted)
    emoji = sum(e.get("emoji_removed", 0) for e in accepted)
    print(f"레코드 {len(result.records)}건, 증거 {len(result.answers['evidence'])}개 → {out_dir}")
    print(f"LLM 시도 {len(attempts)}회 중 검사기 반려 {rejected}회")
    print(f"합격 출력 후처리: 긴 메시지 분할 {split}건, 이모지 제거 {emoji}건")
    return 0


if __name__ == "__main__":
    sys.exit(main())
