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
    sample_background,
    shift_into_period,
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


def _dialogue(dialogue_id: str, day: str, next_day: str | None = None) -> dict:
    """원본과 같은 구조의 배경 대화 하나를 만든다.

    Args:
        dialogue_id: 대화 ID.
        day: 첫 발화 날짜.
        next_day: 마지막 발화 날짜. None이면 같은 날.

    Returns:
        AI Hub 대화 JSON.
    """
    return {
        "header": {
            "dialogueInfo": {"dialogueID": dialogue_id},
            "participantsInfo": [{"participantID": "P01"}, {"participantID": "P02"}],
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
        dialogues = [_dialogue(f"{prefix}-{i}", "2017-11-11", "2017-11-12" if i % 3 == 0 else None)
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


def test_shift_preserves_intervals_and_time_of_day() -> None:
    """날짜만 하루 단위로 옮겨 시간 간격과 시각이 그대로다."""
    base = datetime(2017, 11, 11, 21, 5, tzinfo=KST)
    records = [
        EvidenceRecord(f"r{i}", SourceType.MESSENGER, "x", "t", base + delta, "s", "c", {"k": "v"})
        for i, delta in enumerate([timedelta(0), timedelta(minutes=2), timedelta(hours=11, minutes=25)])
    ]
    shifted, days, over = shift_into_period(records, date(2026, 3, 1), date(2026, 3, 31), random.Random(1))
    assert not over
    assert [r.timestamp - shifted[0].timestamp for r in shifted] == [r.timestamp - base for r in records]
    assert [r.timestamp.time() for r in shifted] == [r.timestamp.time() for r in records]
    assert date(2026, 3, 1) <= shifted[0].timestamp.date() and shifted[-1].timestamp.date() <= date(2026, 3, 31)
    assert shifted[0].timestamp - timedelta(days=days) == base


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
    assert stats["background_threads"] == 6 and stats["synthetic_threads"] == 2


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
