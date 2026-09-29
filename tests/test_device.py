"""가상 기기 조립 테스트.

핵심 불변조건: 에이전트가 보는 ``device/``에는 메시지 출처(AI Hub/합성)나
정답의 흔적이 없고, ``private/``의 정답지는 합성 결과와 같은 내용을 가리킨다.
"""

import json
import random
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from evidence_trace.ingest.device import (
    DEVICE_APP,
    OWNER_ID,
    AssemblyError,
    DeviceConfig,
    allocate_quotas,
    assemble,
    check_leaks,
    load_excluded_dialogues,
    main,
    plan_session_days,
    sample_background,
    write_device,
)
from evidence_trace.ingest.records import Direction, EvidenceRecord, SourceType
from evidence_trace.ingest.synthetic import FewShotPool, generate, load_scenario, write_outputs
from test_synthetic import GOOD, SCENARIO_YAML, FakeModel

KST = timezone(timedelta(hours=9))

BACKGROUND_YAML = """
background:
  source: aihub_sns
  dialogues: 6
  topic_min:
    상거래(쇼핑): 4
  seed: 7
"""


def _dialogue(dialogue_id: str, day: str, next_day: str | None = None, gender: str = "여성") -> dict:
    """원본과 같은 구조의 배경 대화 하나를 만든다.

    Args:
        dialogue_id: 대화 ID.
        day: 첫 발화 날짜.
        next_day: 마지막 발화 날짜. None이면 같은 날.
        gender: 두 참여자의 성별.

    Returns:
        AI Hub 대화 JSON.
    """
    return {
        "header": {
            "dialogueInfo": {"dialogueID": dialogue_id},
            "participantsInfo": [{"participantID": "P01", "gender": gender, "age": "20대"},
                                 {"participantID": "P02", "gender": gender, "age": "20대"}],
        },
        "body": [
            {"utteranceID": "U1", "participantID": "P01", "date": day, "time": "21:05:00", "utterance": "ㅋㅋ 뭐해"},
            {"utteranceID": "U2", "participantID": "P02", "date": day, "time": "21:07:00", "utterance": "#@이름# 입금했어"},
            {"utteranceID": "U3", "participantID": "P01", "date": next_day or day, "time": "08:30:00",
             "utterance": "#@시스템#송금#"},
        ],
    }


@pytest.fixture
def aihub_root(tmp_path: Path) -> Path:
    """주제 두 개(상거래 8개, 시사교육 8개 대화)가 든 가짜 AI Hub 폴더."""
    root = tmp_path / "aihub" / "[라벨]한국어SNS_valid"
    root.mkdir(parents=True)
    for topic, prefix in (("상거래(쇼핑)", "shop"), ("시사교육", "news")):
        dialogues = [_dialogue(f"{prefix}-{i}", "2017-11-11", "2017-11-12" if i % 3 == 0 else None,
                               "여성" if i % 2 else "남성")
                     for i in range(8)]
        name = unicodedata.normalize("NFD", f"{topic}.json")
        (root / name).write_text(json.dumps({"numberOfItems": 8, "data": dialogues}, ensure_ascii=False),
                                 encoding="utf-8")
    return tmp_path / "aihub"


@pytest.fixture
def scenario_file(tmp_path: Path) -> Path:
    """배경 설정이 붙은 테스트 시나리오와 확정된 합성 결과를 만든다."""
    folder = tmp_path / "scenario"
    folder.mkdir()
    path = folder / "scenario.yaml"
    path.write_text(SCENARIO_YAML + BACKGROUND_YAML, encoding="utf-8")
    scenario = load_scenario(path)
    pool = FewShotPool(dialogues=[[("A", "ㅋㅋ")]])
    write_outputs(scenario, generate(scenario, FakeModel(GOOD), pool), folder / "generated")
    return path


@pytest.fixture
def device(scenario_file: Path, aihub_root: Path):
    """조립된 가상 기기."""
    from evidence_trace.ingest.synthetic import load_results

    scenario = load_scenario(scenario_file)
    config = DeviceConfig.from_scenario_file(scenario_file)
    background = sample_background(aihub_root, config, set())
    return scenario, assemble(scenario, load_results(scenario_file.parent / "generated"), background, config)


# ---------- 배경 대화 추출 ----------

def test_allocate_quotas_sums_to_total() -> None:
    """할당량 합은 전체 수와 같고 고정 할당은 그대로다."""
    quotas = allocate_quotas(["a", "b", "c", "d"], 11, {"b": 5})
    assert sum(quotas.values()) == 11 and quotas["b"] == 5


def test_allocate_quotas_rejects_unknown_topic() -> None:
    """데이터에 없는 주제에 할당하면 거부한다."""
    with pytest.raises(AssemblyError, match="없는 주제"):
        allocate_quotas(["a"], 3, {"z": 1})


def test_sample_background_respects_quota_and_exclusion(scenario_file: Path, aihub_root: Path) -> None:
    """주제별 할당량을 지키고 제외 목록의 대화는 뽑지 않는다."""
    config = DeviceConfig.from_scenario_file(scenario_file)
    excluded = {"shop-0", "shop-1"}
    chosen = sample_background(aihub_root, config, excluded)
    topics = [c.topic for c in chosen]
    assert topics.count("상거래(쇼핑)") == 4 and topics.count("시사교육") == 2
    ids = {c.dialogue["header"]["dialogueInfo"]["dialogueID"] for c in chosen}
    assert not ids & excluded


def test_sample_background_is_deterministic(scenario_file: Path, aihub_root: Path) -> None:
    """같은 시드면 같은 대화를 뽑는다."""
    config = DeviceConfig.from_scenario_file(scenario_file)
    first = sample_background(aihub_root, config, set())
    second = sample_background(aihub_root, config, set())
    assert [c.dialogue for c in first] == [c.dialogue for c in second]


def test_sample_background_fails_when_topic_too_small(scenario_file: Path, aihub_root: Path) -> None:
    """할당량을 채울 대화가 부족하면 조용히 넘어가지 않고 거부한다."""
    config = DeviceConfig.from_scenario_file(scenario_file)
    excluded = {f"shop-{i}" for i in range(6)}
    with pytest.raises(AssemblyError, match="할당량"):
        sample_background(aihub_root, config, excluded)


def test_load_excluded_dialogues(tmp_path: Path) -> None:
    """공통 레코드 JSONL에서 대화 ID를 모은다."""
    path = tmp_path / "fewshot.jsonl"
    path.write_text(json.dumps({"source_ref": {"dialogue_id": "d9"}}) + "\n", encoding="utf-8")
    assert load_excluded_dialogues(path) == {"d9"}


# ---------- 조립 ----------

def test_device_passes_leak_check(device) -> None:
    """조립 결과는 누출 검사를 통과한다."""
    scenario, assembled = device
    assert check_leaks(assembled, scenario) == []


def test_answers_point_to_same_synthetic_content(device, scenario_file: Path) -> None:
    """기기 ID 기준 정답지는 합성 결과와 같은 메시지를 가리킨다."""
    from evidence_trace.ingest.synthetic import load_results

    _, assembled = device
    generated = load_results(scenario_file.parent / "generated")
    original = {r.record_id: r.content for r in generated.records}
    on_device = {r.record_id: r.content for r in assembled.records}
    for eid, entry in generated.answers["evidence"].items():
        device_ids = assembled.answers["evidence"][eid]["record_ids"]
        assert [on_device[d] for d in device_ids] == [original[o] for o in entry["record_ids"]]


def test_owner_messages_are_outgoing(device) -> None:
    """소유자가 보낸 메시지만 outgoing이고 나머지는 incoming이다."""
    _, assembled = device
    for record in assembled.records:
        expected = Direction.OUTGOING if record.sender == OWNER_ID else Direction.INCOMING
        assert record.direction is expected
        assert (OWNER_ID in record.recipients) == (record.sender != OWNER_ID)


def test_ids_are_sequential_and_chronological(device) -> None:
    """메시지 번호는 1부터 연속이고 시간순이다."""
    _, assembled = device
    assert [r.record_id for r in assembled.records] == [f"msg-{i:06d}" for i in range(1, len(assembled.records) + 1)]
    times = [r.timestamp for r in assembled.records]
    assert times == sorted(times)


def test_background_moves_into_period(device) -> None:
    """배경 대화 날짜가 사건 기간(3월)으로 옮겨진다."""
    _, assembled = device
    assert all(r.timestamp.year == 2026 and r.timestamp.month == 3 for r in assembled.records)


def test_contact_names_hide_scenario_names(device) -> None:
    """연락처에는 시나리오의 고전 소설 이름이 없고 모두 이름이 붙는다."""
    scenario, assembled = device
    names = {c["name"] for cid, c in assembled.contacts.items() if cid != OWNER_ID}
    assert None not in names
    assert not names & set(scenario.forbidden_names())
    assert assembled.answers["persons"]["victim"]["scenario_name"] == "성춘향"


def test_relation_label_is_kept(tmp_path: Path, aihub_root: Path) -> None:
    """'엄마' 같은 호칭은 연락처 이름으로 그대로 두고 누출로 보지 않는다."""
    from evidence_trace.ingest.synthetic import load_results

    folder = tmp_path / "rel"
    folder.mkdir()
    path = folder / "scenario.yaml"
    path.write_text((SCENARIO_YAML + BACKGROUND_YAML).replace("contact_name: 흥부", "contact_name: 엄마"),
                    encoding="utf-8")
    scenario = load_scenario(path)
    write_outputs(scenario, generate(scenario, FakeModel(GOOD), FewShotPool(dialogues=[[("A", "ㅋ")]])),
                  folder / "generated")
    config = DeviceConfig.from_scenario_file(path)
    assembled = assemble(scenario, load_results(folder / "generated"),
                         sample_background(aihub_root, config, set()), config)
    friend = assembled.answers["persons"]["friend"]
    assert friend["device_name"] == "엄마"
    assert check_leaks(assembled, scenario) == []


@pytest.mark.parametrize(
    ("mutate", "keyword"),
    [
        (lambda d: d.contacts.__setitem__("contact-0001", {"name": "성춘향"}), "성춘향"),
        (lambda d: d.threads.__setitem__("chat-9999", {"participants": ["t_victim"]}), "t_victim"),
        (lambda d: d.threads.__setitem__("chat-9998", {"note": "E01"}), "증거 ID"),
    ],
)
def test_leak_check_catches_injected_leaks(device, mutate, keyword: str) -> None:
    """출처 흔적을 일부러 넣으면 누출 검사가 잡는다."""
    scenario, assembled = device
    mutate(assembled)
    assert any(keyword in p for p in check_leaks(assembled, scenario))


def test_leak_check_catches_wrong_app(device) -> None:
    """앱 이름이 다른 메시지가 섞이면 잡는다."""
    from dataclasses import replace

    scenario, assembled = device
    assembled.records[0] = replace(assembled.records[0], app="aihub_sns")
    problems = check_leaks(assembled, scenario)
    assert any("앱 이름" in p for p in problems)


def test_write_device_separates_public_and_private(device, tmp_path: Path) -> None:
    """정답지와 출처는 private에만 있고 device의 레코드는 무결성 검증을 통과한다."""
    _, assembled = device
    out = tmp_path / "out"
    write_device(assembled, out)
    public = {p.name for p in (out / "device").iterdir()}
    assert public == {"records.jsonl", "contacts.json", "threads.json"}
    assert {p.name for p in (out / "private").iterdir()} == {
        "answers.json", "provenance.jsonl", "stats.json"}
    for line in (out / "device" / "records.jsonl").read_text(encoding="utf-8").splitlines():
        record = EvidenceRecord.from_dict(json.loads(line))
        assert record.app == DEVICE_APP


def test_assembly_is_deterministic(scenario_file: Path, aihub_root: Path) -> None:
    """같은 입력과 시드면 같은 기기가 나온다."""
    from evidence_trace.ingest.synthetic import load_results

    scenario = load_scenario(scenario_file)
    config = DeviceConfig.from_scenario_file(scenario_file)
    generated = load_results(scenario_file.parent / "generated")
    runs = [assemble(scenario, generated, sample_background(aihub_root, config, set()), config) for _ in range(2)]
    assert [r.to_dict() for r in runs[0].records] == [r.to_dict() for r in runs[1].records]
    assert runs[0].contacts == runs[1].contacts


def test_cli_writes_device(scenario_file: Path, aihub_root: Path, tmp_path: Path) -> None:
    """CLI가 누출 검사를 통과한 기기를 저장한다."""
    out = tmp_path / "devices"
    assert main([str(scenario_file), "--aihub", str(aihub_root), "--out", str(out)]) == 0
    stats = json.loads((out / "private" / "stats.json").read_text(encoding="utf-8"))
    assert stats["background_dialogues"] == 6 and stats["synthetic_threads"] == 2


def test_scenario_names_are_unique_on_device(device) -> None:
    """사건 인물의 기기 이름은 다른 어떤 연락처와도 겹치지 않는다."""
    _, assembled = device
    names = [c["name"] for c in assembled.contacts.values()]
    for person in assembled.answers["persons"].values():
        if person["device_name"]:
            assert names.count(person["device_name"]) == 1


def test_name_generator_draw_never_returns_reserved() -> None:
    """예약한 이름은 중복 허용 추출에서도 나오지 않는다."""
    from evidence_trace.ingest.device import NameGenerator

    gen = NameGenerator(random.Random(0))
    reserved = {gen.reserve() for _ in range(50)}
    drawn = {gen.draw() for _ in range(5000)}
    assert not reserved & drawn


# ---------- 여러 날 세션 묶기 ----------

def test_plan_session_days_keeps_sessions_apart() -> None:
    """세션은 기간 안에서 순서대로, 서로 겹치지 않게 배치된다."""
    spans = [0, 2, 0, 1]
    for seed in range(50):
        starts = plan_session_days(spans, date(2026, 3, 1), date(2026, 3, 31), random.Random(seed))
        for (a, span), b in zip(zip(starts, spans), starts[1:]):
            assert (b - a).days >= span + 1
        assert starts[0] >= date(2026, 3, 1)
        assert starts[-1] + timedelta(days=spans[-1]) <= date(2026, 3, 31)


def test_plan_session_days_rejects_overfull_period() -> None:
    """세션이 기간에 다 들어가지 않으면 None을 돌려준다."""
    assert plan_session_days([10, 10, 10], date(2026, 3, 1), date(2026, 3, 20), random.Random(0)) is None


def _background_threads(assembled):
    """조립 결과에서 배경 대화방별 레코드와 출처를 모은다."""
    from collections import defaultdict

    origin = {e["device_record_id"]: e for e in assembled.provenance}
    threads = defaultdict(list)
    for record in assembled.records:
        if origin[record.record_id]["origin"] == "aihub_sns":
            threads[record.thread_id].append((record, origin[record.record_id]))
    return threads


def test_sessions_share_contact_and_profile(device) -> None:
    """한 대화방으로 묶인 세션들은 같은 상대 연락처를 쓰고 성별도 같다."""
    _, assembled = device
    for items in _background_threads(assembled).values():
        dialogues = {p["original_source_ref"]["dialogue_id"] for _, p in items}
        contacts = {r.sender for r, _ in items if r.sender != OWNER_ID}
        assert len(contacts) <= 1
        genders = {"남" if int(d.split("-")[1]) % 2 == 0 else "여" for d in dialogues}
        assert len(genders) == 1, dialogues


def test_sessions_do_not_overlap_in_days(device) -> None:
    """한 대화방의 세션들은 서로 다른 날에 있고, 세션 안 시각은 원본 그대로다."""
    _, assembled = device
    for items in _background_threads(assembled).values():
        by_dialogue = {}
        for record, prov in items:
            by_dialogue.setdefault(prov["original_source_ref"]["dialogue_id"], []).append(record)
        spans = sorted((min(r.timestamp for r in rs), max(r.timestamp for r in rs)) for rs in by_dialogue.values())
        for (_, end_a), (start_b, _) in zip(spans, spans[1:]):
            assert end_a.date() < start_b.date()
        for rs in by_dialogue.values():
            assert {r.timestamp.strftime("%H:%M") for r in rs} == {"21:05", "21:07", "08:30"}


def test_provenance_restores_original_date_and_sender(device) -> None:
    """출처 기록의 날짜 이동 일수와 원래 발신자로 원본을 복원할 수 있다."""
    _, assembled = device
    for items in _background_threads(assembled).values():
        for record, prov in items:
            original_day = record.timestamp.date() - timedelta(days=prov["day_shift"])
            assert original_day.isoformat() in ("2017-11-11", "2017-11-12")
            assert prov["original_sender"].startswith("aihub_sns:")
            assert prov["original_sender"].split(":")[1] == prov["original_source_ref"]["dialogue_id"]


def test_multi_day_ratio_follows_session_distribution(tmp_path: Path) -> None:
    """세션 수 분포대로 묶으면 배경의 여러 날 대화방 비율이 분포에 가까워진다."""
    from evidence_trace.ingest.device import BackgroundDialogue, _background_conversations

    config = DeviceConfig(date(2026, 3, 1), date(2026, 3, 31), 600, {}, 3, {1: 0.3, 2: 0.4, 3: 0.3})
    items = [BackgroundDialogue("t", "t.json", _dialogue(f"x-{i}", "2017-11-11")) for i in range(600)]
    convs, overflow = _background_conversations(items, config, random.Random(1))
    multi = sum(len({r.timestamp.date() for r in c.records}) > 1 for c in convs) / len(convs)
    assert overflow == 0
    assert 0.6 <= multi <= 0.8
    assert sum(len(c.topics) for c in convs) == 600


def test_stats_report_multi_day_ratio(device) -> None:
    """조립 통계에 배경·합성의 여러 날 대화방 비율이 기록된다."""
    _, assembled = device
    ratio = assembled.stats["multi_day_thread_ratio"]
    assert set(ratio) == {"background", "synthetic"}
    assert assembled.stats["background_dialogues"] == 6


def test_invalid_session_distribution_is_rejected(tmp_path: Path) -> None:
    """세션 수 분포가 잘못되면 설정을 거부한다."""
    path = tmp_path / "s.yaml"
    path.write_text(SCENARIO_YAML + BACKGROUND_YAML + "  sessions_per_thread: {0: 1.0}\n", encoding="utf-8")
    with pytest.raises(AssemblyError, match="sessions_per_thread"):
        DeviceConfig.from_scenario_file(path)
