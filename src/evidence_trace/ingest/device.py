"""배경 대화와 합성 증거를 한 대의 가상 휴대폰 메신저 DB로 조립한다.

조립 결과는 두 폴더로 나뉜다. 이 분리가 평가의 공정성을 지키는 장치다.

* ``device/``: 에이전트와 검색 색인이 보는 데이터. 모든 메시지가 같은 앱,
  같은 형식의 ID(``msg-000142``, ``chat-0031``, ``contact-0107``)를 쓰며,
  메시지가 어디서 왔는지(AI Hub인지 합성인지) 알려주는 흔적이 없다.
* ``private/``: 평가할 때만 쓰는 데이터. 기기 ID 기준 정답지, 메시지별 출처
  대응표, 사건 인물 → 기기 연락처 대응표, 조립 통계.

조립 규칙:

* 배경 대화는 AI Hub 원본에서 주제별 할당량만큼 무작위로 뽑는다. 합성 생성기의
  말투 예시로 쓴 대화는 제외한다.
* 배경 대화마다 참여자 한 명을 기기 소유자로 지정한다.
* 배경 대화의 날짜는 사건 기간 안으로 하루 단위로만 옮긴다. 대화 안의 시간
  간격과 하루 중 시각 분포가 그대로 유지된다.
* 연락처 이름은 사건 인물과 배경 인물 모두 같은 이름 생성기로 붙인다. 실제
  휴대폰에도 흔한 호칭("엄마" 등)만 그대로 둔다. 사건 인물의 이름은 기기 안에서
  유일하게 예약하고, 배경 연락처는 나머지 이름에서 중복을 허용해 뽑는다(실제
  연락처에도 동명이인이 흔하다).
* 대화방·연락처 번호는 섞어서 매기고, 같은 분에 온 메시지도 무작위로 정렬한다.
  번호 순서로 합성 대화를 알아챌 수 없게 하기 위해서다.
* 저장 전에 :func:`check_leaks`로 출처 흔적을 검사하고, 하나라도 있으면 저장하지 않는다.

알려진 단순화: AI Hub 대화 하나를 대화방 하나로 둔다. 실제 메신저는 같은 상대와의
대화가 한 방에 쌓이지만, 서로 다른 실제 인물의 대화를 한 방에 이어 붙이면 더
부자연스러워지기 때문이다.

Example:
    프로젝트 루트에서 실행한다::

        $ python -m evidence_trace.ingest.device \\
            data/scenarios/fraud_case_01/scenario.yaml \\
            --aihub data/raw/aihub_sns/extracted/valid \\
            --exclude data/processed/aihub_valid_stratified.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from evidence_trace.ingest.aihub_adapter import DATASET_NAME as AIHUB_DATASET
from evidence_trace.ingest.aihub_adapter import dialogue_to_records, iter_dialogues
from evidence_trace.ingest.records import Direction, EvidenceRecord, SourceType
from evidence_trace.ingest.synthetic import DATASET_NAME as SYNTHETIC_DATASET
from evidence_trace.ingest.synthetic import GenerationResult, Scenario, load_results, load_scenario

DEVICE_APP = "messenger"
"""str: 기기 안 모든 메시지의 앱 이름."""

DEVICE_DB = "messenger.db"
"""str: ``source_ref``에 기록하는 가상 메신저 DB 파일 이름."""

OWNER_ID = "owner"
"""str: 기기 소유자의 발신자 ID."""

RELATION_LABELS = frozenset({"엄마", "아빠", "어머니", "아버지", "누나", "언니", "형", "오빠", "동생"})
"""frozenset[str]: 실제 휴대폰에도 흔해서 연락처 이름으로 그대로 두는 호칭."""

SURNAMES = (
    "김 이 박 최 정 강 조 윤 장 임 한 오 서 신 권 황 안 송 류 전 고 문 양 손 배 백 허 유 남 노"
).split()
"""tuple[str, ...]: 이름 생성기의 성씨 목록."""

GIVEN_NAMES = (
    "민준 서준 도윤 예준 시우 하준 주원 지호 지후 준우 준서 건우 도현 현우 지훈 우진 선우 서진 "
    "민재 현준 연우 유준 정우 승우 승현 시윤 준혁 은우 지환 승민 서연 서윤 지우 서현 민서 하은 "
    "하윤 윤서 지유 지민 채원 지원 수아 다은 은서 예은 수빈 지아 소율 예린 민지 수연 지현 유진 "
    "혜진 은지 현정 미영 은영 지은 성민 성호 영호 동현 상우 재현 태윤 경민 진우 영수 정민 수정 "
    "미경 정희 혜원 가은 나연 다인 보람 소연 아름 유나 은비 재윤 태희 하늘 한결 해진 준영 재민 "
    "동욱 기현 창민 석진 규리 세영 은정 수현 진희 민호"
).split()
"""tuple[str, ...]: 이름 생성기의 이름 목록."""


class AssemblyError(RuntimeError):
    """조립에 필요한 조건이 충족되지 않을 때 발생한다."""


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceConfig:
    """시나리오 파일에서 읽은 조립 설정.

    Attributes:
        period_start: 사건 기간 첫날.
        period_end: 사건 기간 마지막 날.
        dialogues: 배경 대화 수.
        topic_min: 주제 → 최소 대화 수.
        seed: 조립 난수 시드.
    """

    period_start: date
    period_end: date
    dialogues: int
    topic_min: dict[str, int]
    seed: int

    @classmethod
    def from_scenario_file(cls, path: Path) -> DeviceConfig:
        """시나리오 YAML의 ``period``와 ``background`` 항목을 읽는다.

        Args:
            path: 시나리오 YAML 경로.

        Returns:
            조립 설정.

        Raises:
            AssemblyError: 필요한 항목이 없거나 값이 올바르지 않은 경우.
        """
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            period = data["period"]
            background = data["background"]
            start = date.fromisoformat(str(period["start"]))
            end = date.fromisoformat(str(period["end"]))
            topic_min = {
                unicodedata.normalize("NFC", str(k)): int(v)
                for k, v in (background.get("topic_min") or {}).items()
            }
            config = cls(start, end, int(background["dialogues"]), topic_min, int(background.get("seed", 0)))
        except (KeyError, TypeError, ValueError) as exc:
            raise AssemblyError(f"시나리오의 period/background 항목을 읽을 수 없습니다: {exc}") from exc
        if config.period_end < config.period_start:
            raise AssemblyError("사건 기간의 끝이 시작보다 빠릅니다")
        if sum(config.topic_min.values()) > config.dialogues:
            raise AssemblyError("주제별 최소 대화 수의 합이 전체 배경 대화 수보다 큽니다")
        return config


# ---------------------------------------------------------------------------
# 배경 대화 추출
# ---------------------------------------------------------------------------


def allocate_quotas(topics: list[str], total: int, topic_min: dict[str, int]) -> dict[str, int]:
    """주제별로 뽑을 대화 수를 정한다.

    ``topic_min``에 적힌 주제는 그 수만큼, 나머지 주제는 남은 수를 고르게
    나눈다. 나누어떨어지지 않는 몫은 이름순으로 앞 주제부터 하나씩 더한다.

    Args:
        topics: 주제 이름 목록.
        total: 전체 대화 수.
        topic_min: 주제 → 고정 할당량.

    Returns:
        주제 → 할당량. 합은 ``total``과 같다.

    Raises:
        AssemblyError: ``topic_min``에 없는 주제가 있거나, 나눌 주제가 없는데
            남는 수가 있는 경우.

    Example:
        >>> allocate_quotas(["가", "나", "다"], 10, {"가": 4})
        {'가': 4, '나': 3, '다': 3}
    """
    unknown = set(topic_min) - set(topics)
    if unknown:
        raise AssemblyError(f"데이터에 없는 주제: {sorted(unknown)}")
    quotas = dict(topic_min)
    rest = sorted(t for t in topics if t not in topic_min)
    remaining = total - sum(topic_min.values())
    if not rest:
        if remaining:
            raise AssemblyError("고정 할당 외에 나눌 주제가 없습니다")
        return quotas
    base, extra = divmod(remaining, len(rest))
    for i, topic in enumerate(rest):
        quotas[topic] = base + (1 if i < extra else 0)
    return {t: quotas[t] for t in sorted(quotas)}


def load_excluded_dialogues(path: Path | None) -> set[str]:
    """제외할 AI Hub 대화 ID를 공통 레코드 JSONL에서 모은다.

    Args:
        path: 말투 예시로 쓴 공통 레코드 JSONL. None이면 빈 집합.

    Returns:
        대화 ID 집합.
    """
    if path is None:
        return set()
    excluded: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            excluded.add(json.loads(line)["source_ref"]["dialogue_id"])
    return excluded


@dataclass(frozen=True)
class BackgroundDialogue:
    """뽑힌 배경 대화 하나.

    Attributes:
        topic: 주제(원본 파일 이름).
        source_file: 데이터셋 루트 기준 원본 파일 상대 경로(NFC).
        dialogue: 원본 대화 JSON.
    """

    topic: str
    source_file: str
    dialogue: dict[str, Any]


def sample_background(
    root: Path, config: DeviceConfig, excluded: set[str]
) -> list[BackgroundDialogue]:
    """AI Hub 원본에서 주제별 할당량만큼 대화를 무작위로 뽑는다.

    파일마다 저수지 표본 추출(reservoir sampling)을 써서 수 GB 원본을 한 번만
    훑고도 균등한 무작위 표본을 얻는다. 주제마다 시드를 따로 두어 한 주제의
    할당량을 바꿔도 다른 주제의 표본은 그대로다.

    Args:
        root: 압축을 푼 AI Hub 데이터 폴더.
        config: 조립 설정.
        excluded: 뽑지 않을 대화 ID.

    Returns:
        뽑힌 대화 목록(주제 이름순, 주제 안에서는 원본 순서).

    Raises:
        AssemblyError: 폴더에 JSON이 없거나, 어떤 주제의 대화 수가 할당량보다 적은 경우.
    """
    files = {
        unicodedata.normalize("NFC", p.stem): p for p in sorted(root.rglob("*.json"))
    }
    if not files:
        raise AssemblyError(f"AI Hub JSON 파일이 없습니다: {root}")
    quotas = allocate_quotas(list(files), config.dialogues, config.topic_min)

    chosen: list[BackgroundDialogue] = []
    for topic in sorted(files):
        path, quota = files[topic], quotas[topic]
        if quota == 0:
            continue
        rng = random.Random(f"{config.seed}:background:{topic}")
        rel = unicodedata.normalize("NFC", path.relative_to(root).as_posix())
        reservoir: list[tuple[int, dict[str, Any]]] = []
        seen = 0
        for position, dialogue in enumerate(iter_dialogues(path)):
            if str(dialogue["header"]["dialogueInfo"]["dialogueID"]) in excluded:
                continue
            seen += 1
            if len(reservoir) < quota:
                reservoir.append((position, dialogue))
            else:
                slot = rng.randrange(seen)
                if slot < quota:
                    reservoir[slot] = (position, dialogue)
        if len(reservoir) < quota:
            raise AssemblyError(f"{topic}: 대화 {len(reservoir)}개로 할당량 {quota}개를 채울 수 없습니다")
        reservoir.sort(key=lambda item: item[0])
        chosen.extend(BackgroundDialogue(topic, rel, d) for _, d in reservoir)
    return chosen


def shift_into_period(
    records: list[EvidenceRecord], start: date, end: date, rng: random.Random
) -> tuple[list[EvidenceRecord], int, bool]:
    """대화의 날짜를 사건 기간 안으로 하루 단위로 옮긴다.

    모든 메시지를 같은 일수만큼 옮기므로 대화 안의 시간 간격과 하루 중 시각이
    그대로 유지된다. 대화 길이가 사건 기간보다 길면 첫날을 기간 시작에 맞추고
    넘치는 부분은 그대로 둔다.

    Args:
        records: 한 대화의 레코드들.
        start: 사건 기간 첫날.
        end: 사건 기간 마지막 날.
        rng: 재현 가능한 난수 생성기.

    Returns:
        옮긴 레코드 목록, 옮긴 일수(원래 날짜 복원용), 기간을 넘쳤는지 여부.
    """
    first = min(r.timestamp.date() for r in records)
    last = max(r.timestamp.date() for r in records)
    span = (last - first).days
    room = (end - start).days - span
    new_first = start + timedelta(days=rng.randint(0, room)) if room >= 0 else start
    offset = timedelta(days=(new_first - first).days)
    shifted = [replace(r, timestamp=r.timestamp + offset) for r in records]
    return shifted, offset.days, room < 0


# ---------------------------------------------------------------------------
# 연락처 이름
# ---------------------------------------------------------------------------


class NameGenerator:
    """평범한 한국어 이름을 재현 가능하게 만든다.

    :meth:`reserve`로 뽑은 이름은 기기 안에서 유일하고, :meth:`draw`는 예약되지
    않은 이름 중에서 중복을 허용해 뽑는다.

    Attributes:
        rng: 난수 생성기.
        pool: 아직 예약되지 않은 이름 목록.
    """

    def __init__(self, rng: random.Random, exclude: frozenset[str] = frozenset()) -> None:
        """이름 생성기를 만든다.

        Args:
            rng: 재현 가능한 난수 생성기.
            exclude: 만들면 안 되는 이름 (예: 시나리오의 고전 소설 이름).
        """
        self.rng = rng
        self.pool = [s + g for s in SURNAMES for g in GIVEN_NAMES if s + g not in exclude]
        self.rng.shuffle(self.pool)

    def reserve(self) -> str:
        """기기 안에서 유일한 이름을 뽑아 예약한다.

        예약한 이름은 이후 :meth:`reserve`와 :meth:`draw` 어느 쪽에서도 나오지 않는다.

        Returns:
            예약된 이름.

        Raises:
            AssemblyError: 예약할 이름이 남지 않은 경우.
        """
        if not self.pool:
            raise AssemblyError("예약할 수 있는 이름이 남지 않았습니다")
        return self.pool.pop()

    def draw(self) -> str:
        """예약되지 않은 이름 중 하나를 중복을 허용해 뽑는다.

        Returns:
            이름. 앞서 :meth:`draw`로 나온 이름과 같을 수 있다.

        Raises:
            AssemblyError: 뽑을 이름이 남지 않은 경우.
        """
        if not self.pool:
            raise AssemblyError("뽑을 수 있는 이름이 남지 않았습니다")
        return self.rng.choice(self.pool)


# ---------------------------------------------------------------------------
# 조립
# ---------------------------------------------------------------------------


@dataclass
class Conversation:
    """조립 전의 대화방 하나.

    Attributes:
        origin: ``"aihub_sns"`` 또는 ``"synthetic"``.
        original_thread: 원본 ``thread_id``.
        records: 원본 레코드 (배경 대화는 날짜를 옮긴 뒤).
        owner: 이 대화에서 기기 소유자인 원본 발신자 ID.
        participants: 원본 참여자 ID (소유자 포함).
        topic: 배경 대화의 주제. 합성은 ``None``.
        day_shift: 날짜를 옮긴 일수. 원래 날짜 = 기기 날짜 - ``day_shift``일.
    """

    origin: str
    original_thread: str
    records: list[EvidenceRecord]
    owner: str
    participants: list[str]
    topic: str | None = None
    day_shift: int = 0


@dataclass
class AssembledDevice:
    """조립 결과.

    Attributes:
        records: 기기 레코드 (메시지 번호 = 시간순).
        contacts: 연락처 ID → ``{"name": 이름}``. 소유자 포함.
        threads: 대화방 ID → ``{"participants": [...], "messages": 수}``.
        answers: 기기 ID 기준 정답지.
        provenance: 기기 레코드별 출처 기록.
        stats: 조립 통계.
    """

    records: list[EvidenceRecord]
    contacts: dict[str, dict[str, Any]]
    threads: dict[str, dict[str, Any]]
    answers: dict[str, Any]
    provenance: list[dict[str, Any]]
    stats: dict[str, Any] = field(default_factory=dict)


def _synthetic_conversations(scenario: Scenario, generated: GenerationResult) -> list[Conversation]:
    """합성 결과를 대화방 단위로 묶는다.

    Args:
        scenario: 시나리오.
        generated: 확정된 합성 결과.

    Returns:
        합성 대화방 목록 (시나리오 순서).
    """
    grouped: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in generated.records:
        grouped[record.thread_id].append(record)
    owner = f"{SYNTHETIC_DATASET}:{scenario.id}:{scenario.owner}"
    conversations = []
    for thread in scenario.threads:
        key = f"{SYNTHETIC_DATASET}:{scenario.id}:{thread.id}"
        if key not in grouped:
            raise AssemblyError(f"합성 결과에 대화방 {thread.id}가 없습니다")
        participants = [f"{SYNTHETIC_DATASET}:{scenario.id}:{p}" for p in thread.participants]
        conversations.append(Conversation(SYNTHETIC_DATASET, key, grouped[key], owner, participants))
    return conversations


def _background_conversations(
    background: list[BackgroundDialogue], config: DeviceConfig, rng: random.Random
) -> tuple[list[Conversation], int]:
    """배경 대화를 레코드로 바꾸고 소유자를 정한 뒤 날짜를 옮긴다.

    Args:
        background: 뽑힌 배경 대화.
        config: 조립 설정.
        rng: 재현 가능한 난수 생성기.

    Returns:
        배경 대화방 목록과, 사건 기간을 넘친 대화 수.
    """
    conversations: list[Conversation] = []
    overflow = 0
    for item in background:
        records = dialogue_to_records(item.dialogue, item.source_file)
        if not records:
            continue
        participants = sorted({r.sender for r in records} | {p for r in records for p in r.recipients})
        owner = rng.choice(participants)
        shifted, days, over = shift_into_period(records, config.period_start, config.period_end, rng)
        overflow += over
        conversations.append(
            Conversation(AIHUB_DATASET, records[0].thread_id, shifted, owner, participants,
                         item.topic, days)
        )
    return conversations, overflow


def assemble(
    scenario: Scenario,
    generated: GenerationResult,
    background: list[BackgroundDialogue],
    config: DeviceConfig,
) -> AssembledDevice:
    """합성 대화와 배경 대화를 한 대의 가상 기기로 조립한다.

    Args:
        scenario: 시나리오.
        generated: 확정된 합성 결과.
        background: 뽑힌 배경 대화.
        config: 조립 설정.

    Returns:
        조립 결과. 저장 전에 :func:`check_leaks`로 검사해야 한다.

    Raises:
        AssemblyError: 합성 결과가 시나리오와 맞지 않는 경우.
    """
    rng = random.Random(f"{config.seed}:assemble:{scenario.id}")
    synthetic = _synthetic_conversations(scenario, generated)
    backdrop, overflow = _background_conversations(background, config, rng)
    conversations = synthetic + backdrop

    # 연락처: 소유자는 하나로 합치고, 나머지는 섞은 뒤 번호를 매긴다.
    owners = {c.owner for c in conversations}
    others = sorted({p for c in conversations for p in c.participants if p != c.owner})
    rng.shuffle(others)
    contact_of = {o: OWNER_ID for o in owners}
    contact_of.update({p: f"contact-{i:04d}" for i, p in enumerate(others, start=1)})

    names = NameGenerator(rng, exclude=scenario.forbidden_names())
    synthetic_prefix = f"{SYNTHETIC_DATASET}:{scenario.id}:"
    contacts: dict[str, dict[str, Any]] = {OWNER_ID: {"name": None}}
    persons: dict[str, dict[str, Any]] = {}
    # 사건 인물 이름을 먼저 유일하게 예약한 뒤 배경 연락처 이름을 뽑는다.
    scenario_contacts = sorted((p for p in others if p.startswith(synthetic_prefix)), key=contact_of.get)
    for original in scenario_contacts:
        pid = original[len(synthetic_prefix):]
        label = scenario.persons[pid].contact_name
        name = label if label in RELATION_LABELS else names.reserve()
        contacts[contact_of[original]] = {"name": name}
        persons[pid] = {"contact_id": contact_of[original], "device_name": name, "scenario_name": label}
    for original in sorted((p for p in others if not p.startswith(synthetic_prefix)), key=contact_of.get):
        contacts[contact_of[original]] = {"name": names.draw()}
    contacts = {cid: contacts[cid] for cid in sorted(contacts, key=lambda c: (c != OWNER_ID, c))}
    persons[scenario.owner] = {"contact_id": OWNER_ID, "device_name": None,
                               "scenario_name": scenario.persons[scenario.owner].name}

    # 대화방: 섞은 뒤 번호를 매긴다.
    order = list(range(len(conversations)))
    rng.shuffle(order)
    thread_of = {conversations[i].original_thread: f"chat-{n:04d}" for n, i in enumerate(order, start=1)}

    # 메시지: 시간순, 같은 분이면 무작위 순서.
    tagged = [(r.timestamp, rng.random(), c, r) for c in conversations for r in c.records]
    tagged.sort(key=lambda item: (item[0], item[1]))

    records: list[EvidenceRecord] = []
    provenance: list[dict[str, Any]] = []
    device_id_of: dict[str, str] = {}
    thread_members: dict[str, set[str]] = defaultdict(set)
    thread_counts: Counter[str] = Counter()
    for rowid, (_, _, conv, original) in enumerate(tagged, start=1):
        sender = contact_of[original.sender] if original.sender != conv.owner else OWNER_ID
        recipients = tuple(
            OWNER_ID if p == conv.owner else contact_of[p] for p in original.recipients
        )
        thread_id = thread_of[conv.original_thread]
        record = EvidenceRecord(
            record_id=f"msg-{rowid:06d}",
            source_type=SourceType.MESSENGER,
            app=DEVICE_APP,
            thread_id=thread_id,
            timestamp=original.timestamp,
            sender=sender,
            content=original.content,
            source_ref={"db": DEVICE_DB, "table": "messages", "rowid": str(rowid)},
            kind=original.kind,
            recipients=recipients,
            direction=Direction.OUTGOING if sender == OWNER_ID else Direction.INCOMING,
        )
        records.append(record)
        device_id_of[original.record_id] = record.record_id
        thread_members[thread_id].update({sender, *recipients})
        thread_counts[thread_id] += 1
        provenance.append({
            "device_record_id": record.record_id,
            "origin": conv.origin,
            "original_record_id": original.record_id,
            "original_source_ref": original.source_ref,
            "day_shift": conv.day_shift,
        })

    threads = {
        tid: {"participants": sorted(thread_members[tid]), "messages": thread_counts[tid]}
        for tid in sorted(thread_members)
    }

    answers = {
        "scenario_id": scenario.id,
        "split": scenario.split,
        "evidence": {
            eid: {
                "record_ids": [device_id_of[r] for r in e["record_ids"]],
                "thread": thread_of[f"{SYNTHETIC_DATASET}:{scenario.id}:{e['thread']}"],
                "must_convey": e["must_convey"],
                "tags": e["tags"],
            }
            for eid, e in generated.answers["evidence"].items()
        },
        "decoys": {
            thread_of[f"{SYNTHETIC_DATASET}:{scenario.id}:{name}"]: [device_id_of[r] for r in ids]
            for name, ids in generated.answers["decoys"].items()
        },
        "synthetic_threads": {
            thread_of[c.original_thread]: c.original_thread.rsplit(":", 1)[-1] for c in synthetic
        },
        "persons": persons,
    }

    stats = {
        "messages": len(records),
        "synthetic_messages": sum(len(c.records) for c in synthetic),
        "background_messages": sum(len(c.records) for c in backdrop),
        "threads": len(threads),
        "synthetic_threads": len(synthetic),
        "background_threads": len(backdrop),
        "contacts": len(contacts) - 1,
        "background_topics": dict(sorted(Counter(c.topic for c in backdrop).items())),
        "background_overflow_dialogues": overflow,
        "first_message": records[0].timestamp.isoformat() if records else None,
        "last_message": records[-1].timestamp.isoformat() if records else None,
        "seed": config.seed,
    }
    return AssembledDevice(records, contacts, threads, answers, provenance, stats)


# ---------------------------------------------------------------------------
# 누출 검사
# ---------------------------------------------------------------------------

_RECORD_ID = re.compile(r"msg-\d{6}")
_THREAD_ID = re.compile(r"chat-\d{4}")
_PARTY_ID = re.compile(rf"{OWNER_ID}|contact-\d{{4}}")
_EVIDENCE_ID = re.compile(r"\b(?:E\d{2}|INJ\d{2})\b")


def check_leaks(device: AssembledDevice, scenario: Scenario) -> list[str]:
    """에이전트가 볼 ``device/`` 데이터에 출처나 정답의 흔적이 있는지 검사한다.

    검사 항목:

    * 메시지·대화방·발신자·수신자 ID와 앱 이름, ``source_ref``가 정해진 형식인지
    * 메타데이터와 연락처에 데이터셋 이름, 시나리오 ID, 대화방 이름, 인물 ID,
      증거 ID, 시나리오의 고전 소설 이름이 없는지
    * 메시지 번호가 시간순인지

    Args:
        device: 조립 결과.
        scenario: 시나리오. 금지 단어 목록을 만드는 데 쓴다.

    Returns:
        위반 사유 목록. 비어 있으면 통과다.
    """
    problems: list[str] = []
    for record in device.records:
        where = record.record_id
        if not _RECORD_ID.fullmatch(record.record_id):
            problems.append(f"{where}: 메시지 ID 형식")
        if not _THREAD_ID.fullmatch(record.thread_id):
            problems.append(f"{where}: 대화방 ID 형식")
        if not _PARTY_ID.fullmatch(record.sender) or not all(
            _PARTY_ID.fullmatch(r) for r in record.recipients
        ):
            problems.append(f"{where}: 발신자·수신자 ID 형식")
        if record.app != DEVICE_APP:
            problems.append(f"{where}: 앱 이름 {record.app!r}")
        if set(record.source_ref) != {"db", "table", "rowid"} or record.source_ref["db"] != DEVICE_DB:
            problems.append(f"{where}: source_ref 형식")
    for earlier, later in zip(device.records, device.records[1:]):
        if later.timestamp < earlier.timestamp:
            problems.append(f"{later.record_id}: 메시지 번호가 시간순이 아님")
            break

    metadata = json.dumps(
        {
            "meta": [
                {k: v for k, v in r.to_dict().items() if k not in ("content", "sha256")}
                for r in device.records
            ],
            "contacts": device.contacts,
            "threads": device.threads,
        },
        ensure_ascii=False,
    )
    tokens = {AIHUB_DATASET, SYNTHETIC_DATASET, "aihub", scenario.id}
    tokens |= {t.id for t in scenario.threads}
    tokens |= {p for p in scenario.persons if p != OWNER_ID}
    tokens |= set(scenario.forbidden_names())
    tokens -= RELATION_LABELS
    for token in sorted(tokens):
        if token and token in metadata:
            problems.append(f"device 메타데이터에 출처 흔적 {token!r}")
    if _EVIDENCE_ID.search(metadata):
        problems.append(f"device 메타데이터에 증거 ID {_EVIDENCE_ID.search(metadata).group()!r}")
    return problems


# ---------------------------------------------------------------------------
# 저장
# ---------------------------------------------------------------------------


def write_device(device: AssembledDevice, out_dir: Path) -> None:
    """조립 결과를 ``device/``와 ``private/``로 나눠 저장한다.

    Args:
        device: 누출 검사를 통과한 조립 결과.
        out_dir: 저장할 폴더. 원본 대화가 들어 있으므로 git에 올리지 않는 곳이어야 한다.
    """
    public, private = out_dir / "device", out_dir / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    with (public / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in device.records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    for path, payload in (
        (public / "contacts.json", device.contacts),
        (public / "threads.json", device.threads),
        (private / "answers.json", device.answers),
        (private / "stats.json", device.stats),
    ):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (private / "provenance.jsonl").open("w", encoding="utf-8") as handle:
        for entry in device.provenance:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    """명령행 진입점. 가상 기기를 조립하고 누출 검사를 통과하면 저장한다.

    Args:
        argv: 명령행 인자 목록. None이면 ``sys.argv[1:]``을 쓴다.

    Returns:
        프로세스 종료 코드. 성공하면 0, 실패하면 1.
    """
    parser = argparse.ArgumentParser(description="배경 대화 + 합성 증거 → 가상 기기")
    parser.add_argument("scenario", type=Path, help="scenario.yaml 경로")
    parser.add_argument("--aihub", type=Path, required=True, help="압축을 푼 AI Hub 폴더")
    parser.add_argument("--exclude", type=Path, default=None, help="배경에서 뺄 대화가 든 공통 레코드 JSONL")
    parser.add_argument("--generated", type=Path, default=None,
                        help="확정된 합성 결과 폴더 (기본: 시나리오 폴더/generated)")
    parser.add_argument("--out", type=Path, default=None,
                        help="저장 폴더 (기본: data/processed/devices/<시나리오 ID>)")
    args = parser.parse_args(argv)

    try:
        scenario = load_scenario(args.scenario)
        config = DeviceConfig.from_scenario_file(args.scenario)
        generated = load_results(args.generated or args.scenario.parent / "generated")
        print("배경 대화 추출 중 (원본 전체를 한 번 훑어서 몇 분 걸릴 수 있어요)...", flush=True)
        background = sample_background(args.aihub, config, load_excluded_dialogues(args.exclude))
        device = assemble(scenario, generated, background, config)
    except (AssemblyError, FileNotFoundError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    problems = check_leaks(device, scenario)
    if problems:
        print("누출 검사 실패 — 저장하지 않았습니다:", file=sys.stderr)
        for problem in problems[:20]:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    out_dir = args.out or Path("data/processed/devices") / scenario.id
    write_device(device, out_dir)
    s = device.stats
    print(f"메시지 {s['messages']:,d}건 (합성 {s['synthetic_messages']}, 배경 {s['background_messages']:,d})")
    print(f"대화방 {s['threads']:,d}개, 연락처 {s['contacts']:,d}명, 누출 검사 통과 → {out_dir}")
    print(f"배경 주제 분포: {s['background_topics']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
