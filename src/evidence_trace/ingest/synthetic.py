"""사건 시나리오로부터 합성 증거 대화를 생성한다.

역할 분담이 이 모듈의 핵심이다.

* **LLM**은 메시지 문장만 쓴다. 매번 AI Hub 실제 대화 몇 개를 예시로
  받아 말투를 따라 한다.
* **코드**는 시나리오대로 메시지를 배치하고 시각을 정하며, 시스템
  메시지(``#@시스템#송금#`` 등)를 직접 넣는다. 정답지(어느 레코드가
  어느 증거인가)도 코드가 배치한 위치로 기록한다. LLM의 주장은 정답에
  쓰이지 않는다.
* **후처리**는 LLM이 프롬프트만으로는 잘 지키지 못하는 규칙을 결정적으로
  맞춘다. 긴 메시지를 문장부호 위치에서 20자 안팎으로 쪼개 여러 메시지로 보내게
  하고(실제 사람이 끊어 보내는 방식), 문장 끝 마침표·쉼표를 정리하고, 이모지를
  지운다(AI Hub가 원본의 이모티콘을 가린 가공과 같은 방향). 의미를 바꾸지 않는
  규칙만 두므로 검수가 끝난 결과에도 :func:`renormalize_results`로 소급 적용한다.
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

    검수에서 문제가 나온 대화방만 다시 생성해 기존 결과에 합친다::

        $ python -m evidence_trace.ingest.synthetic \\
            data/scenarios/fraud_case_01/scenario.yaml \\
            --fewshot data/processed/aihub_valid_stratified.jsonl \\
            --threads t_accomplice,t_victim_b

    후처리 규칙을 고친 뒤 LLM 없이 기존 결과에만 다시 적용한다::

        $ python -m evidence_trace.ingest.synthetic \\
            data/scenarios/fraud_case_01/scenario.yaml --renormalize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
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

SPLIT_TARGET_CHARS = 20
"""int: 후처리에서 메시지를 끊는 기준 글자 수. 실제 메시지는 평균 약 10자, 90% 지점이
약 19자인데 LLM 출력은 평균 약 22자라, 문장부호 위치에서 이 길이 안팎으로 끊는다."""

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

ONCE_PER_BEAT_MARKERS = frozenset({"#@금융#", "#@전번#", "#@주소#", "#@번호#", "#@URL#", "#@계정#"})
"""frozenset[str]: 한 장면에서 한 번만 쓸 수 있는 정보 표시. 계좌를 세 번 연달아 보내는 식의
부자연스러운 반복을 막는다."""

_MARKER = re.compile(r"#@[^#\s]+#(?:[^#\s]+#)?")
_LATIN_WORD = re.compile(r"[A-Za-z]{3,}")
_UNNATURAL_SYMBOLS = re.compile("[₩$€]")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_EMOJI_JOINERS = re.compile("[\uFE0F\u200D]")

_CHOSEONG_COMPAT = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNGSEONG_COMPAT = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONGSEONG_COMPAT = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
_CONJOINING_TO_COMPAT = {
    **{0x1100 + i: ch for i, ch in enumerate(_CHOSEONG_COMPAT)},
    **{0x1161 + i: ch for i, ch in enumerate(_JUNGSEONG_COMPAT)},
    **{0x11A8 + i: ch for i, ch in enumerate(_JONGSEONG_COMPAT)},
}
"""dict[int, str]: 한글 조합형 자모(U+1100 블록) → 키보드로 치는 일반 자모(U+3131 블록).

실제 AI Hub 메시지의 자모 표현은 99.98%가 일반 자모다. LLM은 "ᅲᅲ"처럼
조합형 자모를 섞어 내는 경우가 있어, 그대로 두면 합성 메시지라는 단서가 된다.
"""
_SPLIT_POINT = re.compile(r"(?<=[.!?~,])\s+")
_SPEAKER_LABELS = "ABCDEFGH"


class ScenarioError(ValueError):
    """시나리오 파일의 내용이 규칙에 맞지 않을 때 발생한다."""


class GenerationError(RuntimeError):
    """재시도 한도 안에 검사기를 통과하는 출력을 얻지 못했을 때 발생한다.

    Attributes:
        log: 실패하기까지의 시도 기록. 실패한 대화방의 반려 사유도 통계에 남기기 위해 둔다.
    """

    def __init__(self, message: str, log: list[dict[str, Any]] | None = None) -> None:
        """예외를 만든다.

        Args:
            message: 오류 메시지.
            log: 실패하기까지의 시도 기록.
        """
        super().__init__(message)
        self.log = log or []


# ---------------------------------------------------------------------------
# 시나리오 모델
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Person:
    """시나리오 등장인물.

    Attributes:
        id: 시나리오 안에서 쓰는 인물 식별자 (예: ``"victim_a"``).
        role: 사건 전체에서의 역할 설명. 문서용이며, LLM에는 대화방별
            역할(:attr:`Thread.roles`)을 보여준다.
        contact_name: 휴대폰 연락처에 저장된 이름. 기기 소유자는 ``None``.
        name: 문서에서 부르는 이름 (예: 기기 소유자 ``"홍길동"``). 본문에
            새어 나오면 안 되는 이름 검사에 함께 쓴다.
    """

    id: str
    role: str
    contact_name: str | None
    name: str | None = None


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
        avoid: 본문에 쓰면 안 되는 단어나 표시. "뻔한 단어 없이 돌려 말하기"
            같은 조건을 LLM에게 부탁만 하지 않고 검사기가 강제하게 한다. chat
            장면에는 같은 대화방의 증거 장면이 쓰는 표시가 자동으로 추가된다.
        opener: chat 장면의 첫 발화자 인물 ID. 지정하면 검사기가 첫 메시지의
            발화자를 확인해 역할이 뒤바뀐 대화를 반려한다.
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
    avoid: tuple[str, ...] = ()
    opener: str | None = None


@dataclass(frozen=True)
class ThreadRole:
    """대화방 안에서 한 인물이 맡는 역할.

    Attributes:
        label: LLM 출력의 발화자 이름으로 쓰는 역할 이름 (예: ``"판매자"``).
            A/B 같은 기호보다 역할 이름을 줄 때 모델이 역할을 덜 헷갈린다.
        description: 이 대화방에서의 역할 설명.
    """

    label: str
    description: str


@dataclass(frozen=True)
class Thread:
    """합성 대화방 하나.

    Attributes:
        id: 대화방 식별자 (예: ``"t_victim_a"``).
        participants: 참여 인물 ID들.
        style: 말투 지시문.
        decoy: 사건과 무관한 오답 후보 대화방이면 True. 오답 후보에는 사건
            개요를 보여주지 않아 평범한 대화가 수상하게 쓰이지 않게 한다.
        beats: 시각 순으로 정렬된 장면들.
        roles: 인물 ID → 이 대화방에서의 역할.
    """

    id: str
    participants: tuple[str, ...]
    style: str
    decoy: bool
    beats: tuple[Beat, ...]
    roles: dict[str, ThreadRole] = field(default_factory=dict)

    def label_of(self, person_id: str) -> str:
        """인물의 발화자 이름(역할 이름)을 돌려준다.

        Args:
            person_id: 인물 ID.

        Returns:
            역할 이름.
        """
        return self.roles[person_id].label


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

    def forbidden_names(self) -> frozenset[str]:
        """본문에 나오면 안 되는 인물 이름을 모은다.

        연락처 이름과 문서용 이름 중, 어느 대화방의 역할 이름("엄마" 등
        일상 호칭)과도 겹치지 않는 것만 포함한다. 실제 데이터에서 이름은
        모두 ``#@이름#``으로 가려져 있으므로, 이름이 그대로 나오면 합성
        메시지라는 단서가 된다.

        Returns:
            금지할 이름 집합.
        """
        labels = {role.label for t in self.threads for role in t.roles.values()}
        names = {n for p in self.persons.values() for n in (p.contact_name, p.name) if n}
        return frozenset(names - labels)

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
    * 대화방의 ``roles``는 모든 참여자를 다뤄야 하고 역할 이름이 겹치면
      안 된다. ``roles``가 없으면 연락처 이름(소유자는 "소유자")을 역할
      이름으로 쓴다.
    * chat의 ``opener``는 그 장면의 발화자여야 한다.

    또한 같은 대화방의 say 장면이 쓰는 표시(예: ``#@금융#``)를 그 대화방
    chat 장면의 ``avoid``에 자동으로 추가한다. 핵심 사실이 정답으로 표시되지
    않은 맥락 대화에 먼저 나오면 정답지가 불완전해지기 때문이다.

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
        p["id"]: Person(
            id=p["id"], role=p["role"], contact_name=p.get("contact_name"), name=p.get("name")
        )
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

        raw_roles = raw_thread.get("roles") or {}
        roles: dict[str, ThreadRole] = {}
        for pid in participants:
            spec = raw_roles.get(pid)
            if spec is None:
                if raw_roles:
                    raise ScenarioError(f"{thread_id}: roles에 참여자 {pid}가 없습니다")
                person = persons[pid]
                roles[pid] = ThreadRole(person.contact_name or "소유자", person.role)
            elif isinstance(spec, dict):
                roles[pid] = ThreadRole(str(spec["label"]), str(spec.get("desc", "")))
            else:
                roles[pid] = ThreadRole(str(spec), persons[pid].role)
        if len({r.label for r in roles.values()}) != len(roles):
            raise ScenarioError(f"{thread_id}: 역할 이름이 겹칩니다")

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
            opener = raw.get("opener")
            if opener is not None and (beat_type != "chat" or opener not in speakers):
                raise ScenarioError(f"{where}: opener는 chat 장면의 발화자여야 합니다")

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
                    avoid=tuple(raw.get("avoid", ())),
                    opener=opener,
                )
            )

        evidence_markers = tuple(dict.fromkeys(m for b in beats if b.type == "say" for m in b.markers))
        if evidence_markers:
            beats = [
                replace(b, avoid=tuple(dict.fromkeys(b.avoid + evidence_markers)))
                if b.type == "chat" else b
                for b in beats
            ]

        threads.append(
            Thread(
                id=thread_id,
                participants=participants,
                style=raw_thread.get("style", ""),
                decoy=bool(raw_thread.get("decoy", False)),
                beats=tuple(beats),
                roles=roles,
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

    def complete(self, system: str, user: str, seed: int, schema: dict[str, Any] | None = None) -> str:
        """프롬프트에 대한 응답 문자열을 돌려준다.

        Args:
            system: 시스템 프롬프트.
            user: 사용자 프롬프트.
            seed: 재현성을 위한 난수 시드.
            schema: 응답에 강제할 JSON 스키마. None이면 기본 스키마.

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

    def complete(self, system: str, user: str, seed: int, schema: dict[str, Any] | None = None) -> str:
        """Ollama에 요청을 보내고 응답 본문을 돌려준다.

        Args:
            system: 시스템 프롬프트.
            user: 사용자 프롬프트.
            seed: 난수 시드.
            schema: 응답에 강제할 JSON 스키마. None이면 :data:`OUTPUT_SCHEMA`.

        Returns:
            모델이 생성한 JSON 문자열.

        Raises:
            GenerationError: 서버에 연결할 수 없거나 응답 형식이 잘못된 경우.
        """
        payload = {
            "model": self.name,
            "stream": False,
            "format": schema or OUTPUT_SCHEMA,
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
"""dict: LLM 출력에 강제하는 기본 JSON 스키마."""


def build_output_schema(
    labels: Iterable[str],
    opener_label: str | None = None,
    message_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """발화자와 메시지 수를 형식으로 제한한 JSON 스키마를 만든다.

    ``opener_label``이 있으면 ``first_message`` 칸을 따로 두고, 그 발화자를
    ``opener_label`` 하나로 제한한다. ``message_range``가 있으면 ``messages``
    배열의 최소·최대 길이를 정한다(첫 메시지 칸이 있으면 그만큼 뺀다). 모델이
    특정 역할로 시작하거나 특정 개수를 고집하는 쏠림은 다시 뽑기로 고쳐지지 않으므로,
    형식 자체로 강제한다. 검사기의 확인은 그대로 두어 이중으로 거른다.

    Args:
        labels: 이 장면에서 말할 수 있는 역할 이름들.
        opener_label: 첫 메시지를 보내야 하는 역할 이름. 없으면 None.
        message_range: 첫 메시지를 포함한 전체 메시지 수의 (최소, 최대). 없으면 제한 없음.

    Returns:
        JSON 스키마.

    Example:
        >>> build_output_schema(["판매자"])["properties"]["messages"]["items"]["properties"]["speaker"]
        {'type': 'string', 'enum': ['판매자']}
        >>> first = build_output_schema(["판매자", "구매자"], "판매자")["properties"]["first_message"]
        >>> first["properties"]["speaker"]
        {'type': 'string', 'enum': ['판매자']}
        >>> items = build_output_schema(["나", "너"], "너", (2, 5))["properties"]["messages"]
        >>> items["minItems"], items["maxItems"]
        (1, 4)
    """
    schema = json.loads(json.dumps(OUTPUT_SCHEMA))
    schema["properties"]["messages"]["items"]["properties"]["speaker"]["enum"] = sorted(labels)
    if opener_label:
        first = {
            "type": "object",
            "properties": {
                "speaker": {"type": "string", "enum": [opener_label]},
                "text": {"type": "string"},
            },
            "required": ["speaker", "text"],
        }
        # 모델은 스키마의 속성 순서대로 쓰므로 첫 메시지를 먼저 두어 대화 순서와 맞춘다.
        schema["properties"] = {"first_message": first, "messages": schema["properties"]["messages"]}
        schema["required"] = ["first_message", "messages"]
    if message_range is not None:
        offset = 1 if opener_label else 0
        low, high = message_range
        schema["properties"]["messages"]["minItems"] = max(0, low - offset)
        schema["properties"]["messages"]["maxItems"] = max(0, high - offset)
    return schema


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
3. '#@시스템#'으로 시작하는 표시는 절대 쓰지 않는다. 위에 나온 표시 말고 다른 '#' 표시를 만들지 않는다.
   가리는 것은 개인정보(이름, 계좌, 전화번호, 주소, 링크, 번호)뿐이다. 금액이나 가게·장소 이름은
   '5만원', '바다 앞 펜션'처럼 그냥 글자로 쓴다.
4. 매 메시지마다 상대 이름을 부르지 않는다. 계좌 같은 정보는 한 번만 보낸다.
5. 영어 단어와 통화 기호(₩ 등)를 쓰지 않는다.
6. speaker에는 [등장인물]에 나온 역할 이름을 그대로 쓰고, 각 역할의 입장을 끝까지 지킨다.
7. 출력은 {"messages": [{"speaker": "역할 이름", "text": "..."}]} 형식의 JSON만 쓴다.\
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
) -> str:
    """장면 하나를 생성하기 위한 사용자 프롬프트를 만든다.

    오답 후보 대화방에는 사건 개요를 넣지 않는다. 평범한 대화가 사건을
    의식해 수상하게 쓰이는 것을 막기 위해서다.

    Args:
        scenario: 사건 시나리오.
        thread: 장면이 속한 대화방. 역할 이름과 설명을 쓴다.
        beat: 생성할 장면.
        history: 이 대화방에서 지금까지 생성된 (역할 이름, 본문) 목록.
        examples: 말투 예시로 보여줄 실제 대화들.

    Returns:
        LLM에 보낼 사용자 프롬프트.
    """
    people = "\n".join(
        f"- {thread.roles[pid].label}: {thread.roles[pid].description}" for pid in thread.participants
    )
    shown = "\n\n".join(f"[예시 {i + 1}]\n{_format_dialogue(d)}" for i, d in enumerate(examples))
    recent = _format_dialogue(history[-12:]) if history else "(대화 시작)"
    low, high = beat.message_range
    count = f"{low}개" if low == high else f"{low}~{high}개"
    speakers = ", ".join(thread.label_of(s) for s in beat.speakers)

    if beat.type == "say":
        task = (
            f"{speakers}가 다음 내용을 전달하는 메시지를 {count} 써라. "
            f"여러 개로 끊어 보내도 된다.\n전달할 내용: {beat.text}"
        )
        if beat.markers:
            task += f"\n반드시 포함할 표시(한 번만): {', '.join(beat.markers)}"
    else:
        task = f"다음 흐름의 대화를 메시지 {count}로 써라. 말하는 사람: {speakers}\n흐름: {beat.text}"
        if beat.opener:
            opener = thread.label_of(beat.opener)
            task += (
                f"\n첫 메시지는 {opener}가 보낸다. first_message에 {opener}의 첫 메시지를, "
                f"messages에 그 뒤에 이어지는 메시지를 쓴다. 메시지 수는 첫 메시지를 포함해 센다."
            )
    if beat.avoid:
        task += f"\n절대 쓰면 안 되는 단어: {', '.join(beat.avoid)}"
    background = "평범한 일상 대화다." if thread.decoy else scenario.summary

    return (
        f"[실제 대화 예시 — 말투만 참고하고 내용은 베끼지 마라]\n{shown}\n\n"
        f"[배경]\n{background}\n\n"
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


def normalize_jamo(text: str) -> str:
    """홀로 쓰인 조합형 자모를 일반 자모로 바꾼다.

    먼저 NFC로 정규화해 음절을 이루는 자모는 완성형 글자로 합친 뒤, 남은
    조합형 자모만 바꾼다. 따라서 완성형 음절은 영향을 받지 않는다.

    Args:
        text: 원래 문자열.

    Returns:
        조합형 자모가 일반 자모로 바뀐 NFC 문자열.

    Example:
        >>> normalize_jamo("다시 보지 말자 \u1172\u1172") == "다시 보지 말자 ㅠㅠ"
        True
        >>> normalize_jamo("휴 ㅋㅋ")
        '휴 ㅋㅋ'
    """
    composed = unicodedata.normalize("NFC", text)
    return composed.translate(_CONJOINING_TO_COMPAT)


def trim_trailing_punct(text: str) -> str:
    """메신저 말투에 맞게 문장 끝 문장부호를 정리한다.

    실제 메시지는 약 17%만 문장부호로 끝나는데 LLM 출력은 약 72%가 그렇다.
    규칙은 다음과 같다.

    * 끝의 쉼표와 단독 마침표는 지운다.
    * 말줄임표("...")와 물음표는 그대로 둔다.
    * 끝의 느낌표 하나는 문장 내용으로 정해지는 절반에서만 지운다. 몇 번을
      실행해도 같은 문장은 같은 결과가 나온다.

    Args:
        text: 원래 메시지.

    Returns:
        정리된 메시지. 정리하면 빈 문자열이 되는 경우 원래 메시지를 돌려준다.

    Example:
        >>> trim_trailing_punct("여기로 주세요.")
        '여기로 주세요'
        >>> trim_trailing_punct("그렇구나...")
        '그렇구나...'
        >>> trim_trailing_punct("언제 와요?")
        '언제 와요?'
    """
    stripped = text.rstrip()
    trimmed = stripped.rstrip(",")
    if trimmed.endswith(".") and not trimmed.endswith(".."):
        trimmed = trimmed[:-1].rstrip()
    if trimmed.endswith("!") and not trimmed.endswith("!!"):
        if hashlib.sha256(trimmed.encode("utf-8")).digest()[0] % 2 == 0:
            trimmed = trimmed[:-1].rstrip()
    return trimmed or stripped


def normalize_messages(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    """LLM 출력을 실제 메신저 말투에 맞게 결정적으로 정리한다.

    조합형 자모를 일반 자모로 바꾸고(:func:`normalize_jamo`), 이모지와 이모지
    결합 문자를 지우고, 공백을 정리한 뒤, :data:`SPLIT_TARGET_CHARS`를 넘는
    메시지는 :func:`split_long_text`로 쪼개 같은 발화자의 연속 메시지로 만들고,
    조각마다 문장 끝 문장부호를 정리한다(:func:`trim_trailing_punct`).
    이모지를 지운 뒤 비어 버린 메시지는 버린다.

    Args:
        messages: ``{"speaker": 라벨, "text": 본문}`` 목록.

    Returns:
        정리된 메시지 목록과 ``{"split": 쪼갠 메시지 수, "emoji_removed": 이모지를
        지운 메시지 수, "jamo_fixed": 자모를 고친 메시지 수, "punct_trimmed":
        문장부호를 정리한 조각 수}`` 통계.

    Example:
        >>> out, stats = normalize_messages([{"speaker": "A", "text": "좋아요😊"}])
        >>> out, stats
        ([{'speaker': 'A', 'text': '좋아요'}], {'split': 0, 'emoji_removed': 1, 'jamo_fixed': 0, 'punct_trimmed': 0})
    """
    stats = {"split": 0, "emoji_removed": 0, "jamo_fixed": 0, "punct_trimmed": 0}
    result: list[dict[str, str]] = []
    for message in messages:
        original = str(message.get("text", ""))
        text = normalize_jamo(original)
        if text != unicodedata.normalize("NFC", original):
            stats["jamo_fixed"] += 1
        cleaned = _EMOJI_JOINERS.sub("", _EMOJI.sub("", text))
        if cleaned != text:
            stats["emoji_removed"] += 1
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue
        pieces = split_long_text(cleaned, SPLIT_TARGET_CHARS)
        if len(pieces) > 1:
            stats["split"] += 1
        for piece in pieces:
            trimmed = trim_trailing_punct(piece)
            if trimmed != piece:
                stats["punct_trimmed"] += 1
            result.append({"speaker": message.get("speaker", ""), "text": trimmed})
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
    forbidden_names: frozenset[str] = frozenset(),
    opener_label: str | None = None,
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
        forbidden_names: 본문에 나오면 안 되는 인물 이름.
        opener_label: 첫 메시지를 보내야 하는 역할 이름. None이면 검사하지 않는다.

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
    if opener_label and messages and messages[0].get("speaker") != opener_label:
        problems.append(f"첫 발화자 {messages[0].get('speaker')!r} (지정: {opener_label!r})")

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
        markers = _MARKER.findall(text)
        for marker in markers:
            if marker.startswith("#@시스템#"):
                problems.append(f"{where}: 시스템 표시 사용 {marker}")
            elif marker not in ALLOWED_TEXT_MARKERS and not marker.startswith("#@이모티콘#"):
                problems.append(f"{where}: 알 수 없는 표시 {marker}")
        bare = _MARKER.sub(" ", text)
        if "#" in bare:
            problems.append(f"{where}: 형식이 잘못된 '#' 표시")
        if _LATIN_WORD.search(bare):
            problems.append(f"{where}: 영어 단어 {_LATIN_WORD.search(bare).group()!r}")
        if _UNNATURAL_SYMBOLS.search(bare):
            problems.append(f"{where}: 통화 기호 사용")
        for name in forbidden_names:
            if name in bare:
                problems.append(f"{where}: 가려야 할 이름 노출 {name!r}")
        for word in beat.avoid:
            if word in text:
                problems.append(f"{where}: 금지어 {word!r} 사용")
        if len(text) >= 6 and text in fewshot_lines:
            problems.append(f"{where}: 예시 문장을 그대로 베낌")

    if emoji_messages > MAX_EMOJI_MESSAGES_PER_BEAT:
        problems.append(f"이모지 포함 메시지 {emoji_messages}개 (최대 {MAX_EMOJI_MESSAGES_PER_BEAT}개)")

    joined = " ".join(str(m.get("text", "")) for m in messages)
    for marker in beat.markers:
        if marker not in joined:
            problems.append(f"필수 표시 {marker} 누락")
    for marker in ONCE_PER_BEAT_MARKERS:
        if joined.count(marker) > 1:
            problems.append(f"{marker} {joined.count(marker)}회 반복 (장면당 1회)")
    return problems


def _parse_output(raw: str) -> list[dict[str, str]]:
    """LLM 응답에서 메시지 목록을 꺼낸다.

    ``first_message``가 있으면 목록 맨 앞에 붙인다.

    Args:
        raw: LLM이 돌려준 문자열.

    Returns:
        메시지 목록.

    Raises:
        ValueError: JSON이 아니거나 ``messages`` 목록이 없는 경우.

    Example:
        >>> _parse_output('{"first_message": {"speaker": "판매자", "text": "도착했어요"}, '
        ...               '"messages": [{"speaker": "구매자", "text": "2번 출구요"}]}')
        [{'speaker': '판매자', 'text': '도착했어요'}, {'speaker': '구매자', 'text': '2번 출구요'}]
    """
    data = json.loads(raw)
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list):
        raise ValueError("messages 목록이 없습니다")
    parsed = [m for m in messages if isinstance(m, dict)]
    first = data.get("first_message")
    if isinstance(first, dict):
        parsed.insert(0, first)
    return parsed


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
        answers: 정답지. ``evidence``, ``decoys``, ``generation`` 키를 가진다.
        log: 생성 시도 기록.
    """

    records: list[EvidenceRecord]
    answers: dict[str, Any]
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


def _thread_key(scenario: Scenario, thread: Thread) -> str:
    """대화방의 레코드 ``thread_id``를 만든다.

    Args:
        scenario: 시나리오.
        thread: 대화방.

    Returns:
        ``"synthetic:<시나리오>:<대화방>"`` 형식 문자열.
    """
    return f"{DATASET_NAME}:{scenario.id}:{thread.id}"


def generate(
    scenario: Scenario,
    model: ChatModel,
    fewshot: FewShotPool,
    seed: int = 20260301,
    max_attempts: int = 4,
    examples_per_prompt: int = 4,
    progress: Callable[[str], None] | None = None,
    only_threads: set[str] | None = None,
) -> GenerationResult:
    """시나리오의 합성 대화방을 생성한다.

    대화방마다 독립된 난수 생성기를 쓰므로, 일부 대화방만 다시 생성해도
    나머지 대화방의 결과는 영향을 받지 않는다. 장면마다 LLM 원본의 메시지
    수를 먼저 검사하고, :func:`normalize_messages`로 후처리한 뒤 나머지
    규칙을 검사한다.

    Args:
        scenario: 생성할 시나리오.
        model: 메시지 문장을 쓸 LLM.
        fewshot: 말투 예시 묶음.
        seed: 재현성을 위한 기본 시드.
        max_attempts: 장면 하나당 최대 생성 시도 횟수.
        examples_per_prompt: 프롬프트마다 보여줄 예시 대화 수.
        progress: 진행 상황 문자열을 받을 함수. None이면 출력하지 않는다.
        only_threads: 생성할 대화방 ID. None이면 전부 생성한다.

    Returns:
        생성한 대화방의 레코드, 정답지, 시도 기록.

    Raises:
        ScenarioError: ``only_threads``에 시나리오에 없는 대화방이 있는 경우.
        GenerationError: 어떤 장면이 ``max_attempts`` 안에 검사를 통과하지 못한 경우.
            마지막 시도의 위반 사유가 메시지에 담긴다.
    """
    known = {t.id for t in scenario.threads}
    if only_threads is not None and not only_threads <= known:
        raise ScenarioError(f"시나리오에 없는 대화방: {sorted(only_threads - known)}")
    threads = [t for t in scenario.threads if only_threads is None or t.id in only_threads]

    report = progress or (lambda _message: None)
    total_beats = sum(len(t.beats) for t in threads)
    forbidden = scenario.forbidden_names()
    done = 0
    records: list[EvidenceRecord] = []
    evidence: dict[str, Any] = {}
    decoys: dict[str, list[str]] = {}
    generation: dict[str, Any] = {}
    log: list[dict[str, Any]] = []

    for thread in threads:
        rng = random.Random(f"{seed}:{scenario.id}:{thread.id}")
        by_label = {thread.label_of(pid): pid for pid in thread.participants}
        thread_key = _thread_key(scenario, thread)
        history: list[tuple[str, str]] = []
        thread_record_ids: list[str] = []
        generation[thread.id] = {"model": model.name, "seed": seed}

        for position, beat in enumerate(thread.beats):
            done += 1
            where = f"[{done}/{total_beats}] {thread.id} 장면 {beat.index}"
            following = thread.beats[position + 1].at if position + 1 < len(thread.beats) else None

            if beat.literal is not None:
                messages = [{"speaker": thread.label_of(beat.speakers[0]), "text": beat.literal}]
                log.append({"thread": thread.id, "beat": beat.index, "attempt": 0, "ok": True,
                            "problems": [], "literal": True, "seed": seed})
                report(f"{where}: 고정 문구")
            else:
                allowed = {thread.label_of(s) for s in beat.speakers}
                opener_label = thread.label_of(beat.opener) if beat.opener else None
                schema = build_output_schema(allowed, opener_label, beat.message_range)
                problems: list[str] = []
                messages = []
                for attempt in range(1, max_attempts + 1):
                    prompt = build_user_prompt(
                        scenario, thread, beat, history, fewshot.sample(rng, examples_per_prompt)
                    )
                    raw = model.complete(
                        SYSTEM_PROMPT, prompt,
                        _beat_seed(seed, scenario.id, thread.id, beat.index, attempt),
                        schema,
                    )
                    fixes = {"split": 0, "emoji_removed": 0, "jamo_fixed": 0, "punct_trimmed": 0}
                    try:
                        raw_messages = _parse_output(raw)
                        low, high = beat.message_range
                        if low <= len(raw_messages) <= high:
                            messages, fixes = normalize_messages(raw_messages)
                            problems = validate_messages(
                                messages, beat, allowed, fewshot.lines, check_count=False,
                                forbidden_names=forbidden, opener_label=opener_label,
                            )
                        else:
                            problems = [f"메시지 수 {len(raw_messages)}개 (허용 {low}~{high})"]
                    except (ValueError, json.JSONDecodeError) as exc:
                        problems = [f"출력 파싱 실패: {exc}"]
                    log.append({"thread": thread.id, "beat": beat.index, "attempt": attempt,
                                "ok": not problems, "problems": problems, "seed": seed, **fixes})
                    status = "합격" if not problems else f"반려({problems[0]})"
                    report(f"{where} 시도 {attempt}: {status}")
                    if not problems:
                        break
                else:
                    raise GenerationError(
                        f"{thread.id} 장면 {beat.index}: {max_attempts}회 시도 모두 불합격 — {problems}",
                        log,
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

            if beat.evidence_id:
                evidence[beat.evidence_id] = {
                    "record_ids": beat_ids,
                    "thread": thread.id,
                    "must_convey": beat.text,
                    "tags": list(beat.tags),
                }
        if thread.decoy:
            decoys[thread.id] = thread_record_ids

    records.sort(key=lambda r: (r.timestamp, r.record_id))
    answers = {
        "scenario_id": scenario.id,
        "split": scenario.split,
        "evidence": evidence,
        "decoys": decoys,
        "generation": generation,
    }
    return GenerationResult(records=records, answers=answers, log=log)


def merge_results(previous: GenerationResult, update: GenerationResult) -> GenerationResult:
    """이전 생성 결과에서 일부 대화방을 새 결과로 교체한다.

    ``update``에 포함된 대화방의 레코드·증거·오답 후보·생성 정보를 이전
    결과에서 모두 지우고 새 것으로 바꾼다. 나머지 대화방은 그대로 둔다.

    Args:
        previous: 기존 결과 (디스크에서 읽은 것).
        update: 일부 대화방만 새로 생성한 결과.

    Returns:
        합쳐진 결과. 레코드는 시각 순으로 정렬되고, 시도 기록은 이전 기록 뒤에
        새 기록이 이어진다.
    """
    replaced = set(update.answers["generation"])
    records = [r for r in previous.records if r.source_ref.get("thread") not in replaced]
    records += update.records
    records.sort(key=lambda r: (r.timestamp, r.record_id))

    answers = json.loads(json.dumps(previous.answers))
    for legacy in ("model", "seed"):  # 대화방별 generation 정보로 옮겨간 옛 키
        answers.pop(legacy, None)
    answers["evidence"] = {
        k: v for k, v in answers.get("evidence", {}).items() if v["thread"] not in replaced
    }
    answers["evidence"].update(update.answers["evidence"])
    answers["decoys"] = {k: v for k, v in answers.get("decoys", {}).items() if k not in replaced}
    answers["decoys"].update(update.answers["decoys"])
    answers.setdefault("generation", {}).update(update.answers["generation"])
    return GenerationResult(records=records, answers=answers, log=previous.log + update.log)


@dataclass
class IncrementalOutcome:
    """대화방 단위 생성의 결과.

    Attributes:
        result: 저장된 마지막 결과 (성공한 대화방까지 합쳐진 상태).
        completed: 이번 실행에서 생성·저장에 성공한 대화방 ID (순서대로).
        remaining: 생성하지 못한 대화방 ID (실패한 것 포함, 순서대로).
        error: 멈춘 이유. 모두 성공했으면 None.
        attempts: 이번 실행의 LLM 시도 기록.
    """

    result: GenerationResult
    completed: list[str]
    remaining: list[str]
    error: str | None
    attempts: list[dict[str, Any]]


def generate_incrementally(
    scenario: Scenario,
    model: ChatModel,
    fewshot: FewShotPool,
    previous: GenerationResult,
    thread_ids: list[str],
    out_dir: Path,
    progress: Callable[[str], None] | None = None,
    **options: Any,
) -> IncrementalOutcome:
    """대화방을 하나씩 생성해 기존 결과에 합치고, 하나 끝날 때마다 저장한다.

    어떤 대화방이 시도 한도 안에 검사를 통과하지 못하면 거기서 멈추지만, 그 전에
    성공한 대화방은 이미 저장되어 있다. 대화방마다 독립된 난수를 쓰므로 한꺼번에
    생성한 결과와 같다.

    Args:
        scenario: 시나리오.
        model: 메시지 문장을 쓸 LLM.
        fewshot: 말투 예시 묶음.
        previous: 기존 결과 (디스크에서 읽은 것).
        thread_ids: 생성할 대화방 ID. 시나리오 순서대로 처리한다.
        out_dir: 저장 폴더.
        progress: 진행 상황 문자열을 받을 함수.
        **options: :func:`generate`에 그대로 넘길 인자 (seed, max_attempts 등).

    Returns:
        저장된 결과, 성공·남은 대화방 목록, 멈춘 이유, 이번 실행의 시도 기록.

    Raises:
        ScenarioError: 시나리오에 없는 대화방이 있는 경우.
    """
    known = [t.id for t in scenario.threads]
    unknown = set(thread_ids) - set(known)
    if unknown:
        raise ScenarioError(f"시나리오에 없는 대화방: {sorted(unknown)}")
    ordered = [t for t in known if t in set(thread_ids)]

    current = previous
    completed: list[str] = []
    attempts: list[dict[str, Any]] = []
    for index, thread_id in enumerate(ordered):
        try:
            update = generate(scenario, model, fewshot, only_threads={thread_id}, progress=progress, **options)
        except GenerationError as exc:
            attempts += [e for e in exc.log if not e.get("literal")]
            return IncrementalOutcome(current, completed, ordered[index:], str(exc), attempts)
        attempts += [e for e in update.log if not e.get("literal")]
        current = merge_results(current, update)
        write_outputs(scenario, current, out_dir)
        completed.append(thread_id)
    return IncrementalOutcome(current, completed, [], None, attempts)


def load_results(out_dir: Path) -> GenerationResult:
    """저장된 생성 결과를 읽는다. 레코드는 무결성 검증을 거친다.

    Args:
        out_dir: :func:`write_outputs`가 저장한 폴더.

    Returns:
        읽은 결과.

    Raises:
        FileNotFoundError: ``records.jsonl``이나 ``answers.json``이 없는 경우.
        IntegrityError: 저장된 레코드가 변조된 경우.
    """
    with (out_dir / "records.jsonl").open(encoding="utf-8") as handle:
        records = [EvidenceRecord.from_dict(json.loads(line)) for line in handle]
    answers = json.loads((out_dir / "answers.json").read_text(encoding="utf-8"))
    log_path = out_dir / "generation_log.jsonl"
    log = []
    if log_path.exists():
        with log_path.open(encoding="utf-8") as handle:
            log = [json.loads(line) for line in handle]
    return GenerationResult(records=records, answers=answers, log=log)


def renormalize_results(result: GenerationResult) -> tuple[GenerationResult, dict[str, int]]:
    """이미 만든 결과에 현재의 결정적 후처리를 다시 적용한다.

    LLM을 다시 부르지 않는다. 적용하는 규칙(자모 정규화, 문장부호 정리, 문장부호
    위치에서 끊기)은 모두 의미를 바꾸지 않으므로 검수 결과가 그대로 유효하다.
    고정 문구(literal/event) 메시지는 건드리지 않는다.

    메시지를 끊으면 레코드 수가 늘어나므로 대화방마다 레코드 ID를 순서대로 다시
    매기고, 정답지와 오답 후보 목록의 ID도 새 ID로 바꾼다. 끊어진 조각들은 원래
    메시지와 같은 시각을 가진다.

    Args:
        result: 기존 생성 결과.

    Returns:
        새 결과와 ``{"changed": 내용이 바뀐 원래 메시지 수, "split": 여러 조각으로
        나뉜 메시지 수, "records_before": 이전 레코드 수, "records_after": 이후
        레코드 수}``. 시도 기록 끝에 이번 작업 기록이 추가된다.
    """
    literal_beats = {
        (e["thread"], str(e["beat"])) for e in result.log if e.get("literal")
    }
    by_thread: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in result.records:
        by_thread[record.thread_id].append(record)

    new_ids: dict[str, list[str]] = {}
    records: list[EvidenceRecord] = []
    changed = split = 0
    for thread_key, thread_records in by_thread.items():
        thread_records.sort(key=lambda r: r.record_id)
        counter = 0
        for record in thread_records:
            beat_key = (record.source_ref.get("thread"), record.source_ref.get("beat"))
            if beat_key in literal_beats:
                pieces = [record.content]
            else:
                text = normalize_jamo(record.content)
                pieces = [trim_trailing_punct(p) for p in split_long_text(text, SPLIT_TARGET_CHARS)]
            if pieces != [record.content]:
                changed += 1
            if len(pieces) > 1:
                split += 1
            ids = []
            for piece in pieces:
                new_id = f"{thread_key}:{counter:04d}"
                counter += 1
                records.append(replace(record, record_id=new_id, content=piece, kind=classify_kind(piece)))
                ids.append(new_id)
            new_ids[record.record_id] = ids
    records.sort(key=lambda r: (r.timestamp, r.record_id))

    answers = json.loads(json.dumps(result.answers))
    for entry in answers.get("evidence", {}).values():
        entry["record_ids"] = [n for old in entry["record_ids"] for n in new_ids[old]]
    answers["decoys"] = {
        thread: [n for old in ids for n in new_ids[old]] for thread, ids in answers.get("decoys", {}).items()
    }
    summary = {"changed": changed, "split": split,
               "records_before": len(result.records), "records_after": len(records)}
    log = result.log + [{"action": "renormalize", **summary}]
    return GenerationResult(records=records, answers=answers, log=log), summary


def build_review(scenario: Scenario, result: GenerationResult) -> list[str]:
    """검수용 대화록을 레코드와 정답지로부터 만든다.

    생성 과정이 아니라 저장된 데이터로 만들기 때문에, 일부 대화방만 다시
    생성한 뒤에도 전체 대화록이 일관되게 나온다.

    Args:
        scenario: 시나리오. 대화방 순서와 연락처 이름을 쓴다.
        result: 생성 결과.

    Returns:
        마크다운 줄 목록.
    """
    marks = {rid: eid for eid, e in result.answers["evidence"].items() for rid in e["record_ids"]}
    by_thread: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in result.records:
        by_thread[record.source_ref.get("thread", "")].append(record)

    lines = [f"# {scenario.title} ({scenario.id}) 검수용 대화록", ""]
    for thread in scenario.threads:
        names = ", ".join(scenario.persons[p].contact_name or "기기 소유자" for p in thread.participants)
        lines += [f"## {thread.id}{' (오답 후보)' if thread.decoy else ''} — {names}", ""]
        for record in by_thread.get(thread.id, []):
            person = record.sender.rsplit(":", 1)[-1]
            who = scenario.persons[person].contact_name or "소유자"
            mark = f" **[{marks[record.record_id]}]**" if record.record_id in marks else ""
            lines.append(f"- `{record.timestamp.strftime('%m-%d %H:%M')}` {who}: {record.content}{mark}")
        lines.append("")
    return lines


def write_outputs(scenario: Scenario, result: GenerationResult, out_dir: Path) -> None:
    """생성 결과를 파일로 저장한다.

    Args:
        scenario: 시나리오. 검수용 대화록을 만들 때 쓴다.
        result: 저장할 결과.
        out_dir: 저장할 폴더. 없으면 만든다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in result.records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    (out_dir / "answers.json").write_text(
        json.dumps(result.answers, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "review.md").write_text("\n".join(build_review(scenario, result)) + "\n", encoding="utf-8")
    with (out_dir / "generation_log.jsonl").open("w", encoding="utf-8") as handle:
        for entry in result.log:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    """명령행 진입점. 시나리오를 읽어 합성 대화를 생성하고 저장한다.

    ``--threads``를 주면 지정한 대화방만 다시 생성해 기존 결과에 합친다.
    검수를 통과한 대화방은 그대로 두고 문제 있는 대화방만 고칠 때 쓴다. 이때는
    대화방 하나가 끝날 때마다 저장하므로, 중간에 멈춰도 성공한 대화방은 남고
    남은 대화방 목록이 출력된다. ``--threads`` 없이 전체를 생성할 때는 모든
    대화방이 성공해야만 저장해서, 실패했을 때 기존 결과가 반쯤 덮어써지지 않게 한다.

    Args:
        argv: 명령행 인자 목록. None이면 ``sys.argv[1:]``을 쓴다.

    Returns:
        프로세스 종료 코드. 성공하면 0, 실패하면 1.
    """
    parser = argparse.ArgumentParser(description="사건 시나리오 → 합성 증거 대화")
    parser.add_argument("scenario", type=Path, help="scenario.yaml 경로")
    parser.add_argument("--fewshot", type=Path, default=None, help="말투 예시용 공통 레코드 JSONL")
    parser.add_argument("--model", default="exaone3.5:7.8b", help="Ollama 모델 이름")
    parser.add_argument("--seed", type=int, default=20260301, help="기본 시드")
    parser.add_argument("--max-attempts", type=int, default=6, help="장면당 최대 생성 시도 횟수")
    parser.add_argument("--threads", default=None,
                        help="다시 생성할 대화방 ID를 쉼표로 (예: t_victim_b,t_friend)")
    parser.add_argument("--out", type=Path, default=None, help="저장 폴더 (기본: 시나리오 폴더/generated)")
    parser.add_argument("--renormalize", action="store_true",
                        help="LLM 없이 기존 결과에 현재 후처리 규칙만 다시 적용")
    args = parser.parse_args(argv)
    out_dir = args.out or args.scenario.parent / "generated"
    only = {t.strip() for t in args.threads.split(",") if t.strip()} if args.threads else None

    if args.renormalize:
        try:
            scenario = load_scenario(args.scenario)
            fixed, summary = renormalize_results(load_results(out_dir))
        except (ScenarioError, FileNotFoundError, ValueError) as exc:
            print(f"오류: {exc}", file=sys.stderr)
            return 1
        write_outputs(scenario, fixed, out_dir)
        print(f"후처리 재적용: 메시지 {summary['changed']}건 수정(분할 {summary['split']}건), "
              f"레코드 {summary['records_before']} → {summary['records_after']}건 → {out_dir}")
        return 0
    if args.fewshot is None:
        print("오류: --fewshot이 필요합니다 (--renormalize가 아닐 때)", file=sys.stderr)
        return 1

    report = lambda message: print(message, flush=True)  # noqa: E731
    try:
        scenario = load_scenario(args.scenario)
        pool = FewShotPool.from_jsonl(args.fewshot)
        model = OllamaChatModel(name=args.model)
        if only:
            outcome = generate_incrementally(
                scenario, model, pool, load_results(out_dir), sorted(only), out_dir,
                progress=report, seed=args.seed, max_attempts=args.max_attempts,
            )
            _print_attempt_summary(outcome.attempts)
            print(f"저장 완료 대화방 {len(outcome.completed)}개, "
                  f"레코드 {len(outcome.result.records)}건 → {out_dir}")
            if outcome.error:
                print(f"오류: {outcome.error}", file=sys.stderr)
                print(f"남은 대화방 (이 목록으로 --threads 다시 실행): {','.join(outcome.remaining)}",
                      file=sys.stderr)
                return 1
            return 0
        result = generate(scenario, model, pool, seed=args.seed,
                          max_attempts=args.max_attempts, progress=report)
    except (ScenarioError, GenerationError, FileNotFoundError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    write_outputs(scenario, result, out_dir)
    print(f"레코드 {len(result.records)}건, 증거 {len(result.answers['evidence'])}개 → {out_dir}")
    _print_attempt_summary([e for e in result.log if not e.get("literal")])
    return 0


def _print_attempt_summary(attempts: list[dict[str, Any]]) -> None:
    """LLM 시도 통계와 후처리 통계를 출력한다.

    Args:
        attempts: 고정 문구를 뺀 시도 기록.
    """
    rejected = sum(1 for e in attempts if not e["ok"])
    accepted = [e for e in attempts if e["ok"]]
    split = sum(e.get("split", 0) for e in accepted)
    emoji = sum(e.get("emoji_removed", 0) for e in accepted)
    jamo = sum(e.get("jamo_fixed", 0) for e in accepted)
    punct = sum(e.get("punct_trimmed", 0) for e in accepted)
    print(f"이번 실행: LLM 시도 {len(attempts)}회 중 검사기 반려 {rejected}회")
    print(f"합격 출력 후처리: 긴 메시지 분할 {split}건, 이모지 제거 {emoji}건, "
          f"자모 정규화 {jamo}건, 문장부호 정리 {punct}건")


if __name__ == "__main__":
    sys.exit(main())
