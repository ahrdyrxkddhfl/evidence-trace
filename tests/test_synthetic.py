"""합성 증거 생성기 테스트.

실제 LLM 대신 정해진 응답을 돌려주는 가짜 모델로 생성 규칙을 확인한다.
핵심 불변조건: 정답지는 코드가 배치한 위치로 정해지고, 레코드에는 정답
정보가 새어 들어가지 않는다.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random

import pytest

from evidence_trace.ingest.synthetic import (
    Beat,
    FewShotPool,
    GenerationError,
    ScenarioError,
    generate,
    load_scenario,
    normalize_messages,
    split_long_text,
    schedule_times,
    validate_messages,
    write_outputs,
)

KST = timezone(timedelta(hours=9))

SCENARIO_YAML = """
scenario_id: test_case
title: 테스트 사건
split: dev
timezone: "+09:00"
period: {start: 2026-03-01, end: 2026-03-31}
summary: 테스트용 사건.
persons:
  - {id: suspect, name: 홍길동, role: 피의자, contact_name: null}
  - {id: victim, role: 피해자, contact_name: 성춘향}
  - {id: friend, role: 친구, contact_name: 흥부}
threads:
  - thread: t_victim
    participants: [suspect, victim]
    roles:
      suspect: {label: 판매자, desc: 사기 판매자}
      victim: {label: 구매자, desc: 구매자}
    style: 존댓말
    beats:
      - {type: chat, at: "2026-03-03T19:10", messages: 2-3, intent: 물건 문의}
      - type: say
        evidence_id: E01
        at: "2026-03-03T19:20"
        speaker: suspect
        must_convey: 선입금 요구
        markers: ["#@금융#"]
        tags: [입금요청]
      - {type: say, evidence_id: E02, at: "2026-03-03T19:30", speaker: victim,
         must_convey: 송금, literal: "#@시스템#송금#"}
  - thread: t_friend
    participants: [suspect, friend]
    roles:
      suspect: {label: 얻어먹은친구, desc: 친구}
      friend: {label: 계산한친구, desc: 친구}
    style: 반말
    decoy: true
    beats:
      - {type: chat, at: "2026-03-07T22:10", messages: 2, intent: 더치페이}
      - {type: event, at: "2026-03-07T22:25", speaker: suspect, literal: "#@시스템#송금#"}
"""


class FakeModel:
    """미리 정한 응답을 순서대로 돌려주는 가짜 LLM.

    Attributes:
        name: 모델 이름.
        responses: 돌려줄 응답 목록. 앞에서부터 하나씩 꺼낸다.
        calls: 받은 사용자 프롬프트 기록.
    """

    name = "fake"

    def __init__(self, responses: list[str]) -> None:
        """가짜 모델을 만든다.

        Args:
            responses: 순서대로 돌려줄 응답 문자열 목록.
        """
        self.responses = list(responses)
        self.calls: list[str] = []
        self.schemas: list[dict | None] = []

    def complete(self, system: str, user: str, seed: int, schema: dict | None = None) -> str:
        """다음 응답을 돌려준다.

        Args:
            system: 시스템 프롬프트 (사용하지 않음).
            user: 사용자 프롬프트. 기록한다.
            seed: 시드 (사용하지 않음).
            schema: 응답 스키마. 기록한다.

        Returns:
            미리 정한 다음 응답.
        """
        self.calls.append(user)
        self.schemas.append(schema)
        return self.responses.pop(0)


def _reply(*pairs: tuple[str, str]) -> str:
    """(라벨, 본문) 쌍들로 LLM 응답 JSON을 만든다."""
    return json.dumps({"messages": [{"speaker": s, "text": t} for s, t in pairs]}, ensure_ascii=False)


GOOD = [
    _reply(("구매자", "안녕하세요 아직 있나요"), ("판매자", "네 있어요")),            # t_victim chat
    _reply(("판매자", "직거래는 어렵고요"), ("판매자", "여기로 먼저 #@금융#")),       # E01 say
    _reply(("얻어먹은친구", "밥값 얼마더라"), ("계산한친구", "#@금융# 여기로 ㄱ")),   # t_friend chat
]


@pytest.fixture
def scenario_path(tmp_path: Path) -> Path:
    """테스트용 시나리오 파일을 만든다."""
    path = tmp_path / "scenario.yaml"
    path.write_text(SCENARIO_YAML, encoding="utf-8")
    return path


@pytest.fixture
def pool() -> FewShotPool:
    """예시 대화 하나가 든 예시 묶음."""
    return FewShotPool(dialogues=[[("A", "아 헐"), ("B", "왴ㅋㅋㅋㅋ")]], lines={"아 헐", "실제 대화 예시 문장"})


def test_answers_come_from_code_placement(scenario_path: Path, pool: FewShotPool) -> None:
    """정답지의 레코드 ID는 코드가 배치한 레코드와 정확히 일치한다."""
    result = generate(load_scenario(scenario_path), FakeModel(GOOD), pool)
    by_id = {r.record_id: r for r in result.records}

    e01 = result.answers["evidence"]["E01"]["record_ids"]
    assert [by_id[i].content for i in e01] == ["직거래는 어렵고요", "여기로 먼저 #@금융#"]
    e02 = result.answers["evidence"]["E02"]["record_ids"]
    assert [by_id[i].content for i in e02] == ["#@시스템#송금#"]
    assert set(result.answers["decoys"]) == {"t_friend"}


def test_literal_beats_skip_llm(scenario_path: Path, pool: FewShotPool) -> None:
    """literal 장면은 LLM을 부르지 않고 코드가 그대로 넣는다."""
    model = FakeModel(GOOD)
    generate(load_scenario(scenario_path), model, pool)
    assert len(model.calls) == 3


def test_records_do_not_leak_answers(scenario_path: Path, pool: FewShotPool) -> None:
    """레코드 어디에도 증거 ID나 오답 후보 표시가 없다."""
    result = generate(load_scenario(scenario_path), FakeModel(GOOD), pool)
    dumped = json.dumps([r.to_dict() for r in result.records], ensure_ascii=False)
    assert "E01" not in dumped and "E02" not in dumped and "decoy" not in dumped


def test_invalid_output_is_retried_then_accepted(scenario_path: Path, pool: FewShotPool) -> None:
    """검사기에 걸린 출력은 버리고 다시 생성한다."""
    bad = _reply(("판매자", "여기로 입금 #@시스템#송금#"))
    model = FakeModel([GOOD[0], bad, GOOD[1], GOOD[2]])
    result = generate(load_scenario(scenario_path), model, pool)
    rejected = [e for e in result.log if not e["ok"]]
    assert len(rejected) == 1 and "시스템 표시" in rejected[0]["problems"][0]
    assert result.answers["evidence"]["E01"]["record_ids"]


def test_generation_fails_loudly_after_max_attempts(scenario_path: Path, pool: FewShotPool) -> None:
    """한도까지 불합격이면 조용히 넘어가지 않고 예외를 던진다."""
    bad = _reply(("판매자", "가" * 41))
    with pytest.raises(GenerationError, match="t_victim 장면 0"):
        generate(load_scenario(scenario_path), FakeModel([bad] * 4), pool, max_attempts=4)


def test_timestamps_stay_before_next_beat(scenario_path: Path, pool: FewShotPool) -> None:
    """장면의 메시지는 다음 장면 시작보다 앞선다."""
    result = generate(load_scenario(scenario_path), FakeModel(GOOD), pool)
    by_id = {r.record_id: r for r in result.records}
    last_e01 = max(by_id[i].timestamp for i in result.answers["evidence"]["E01"]["record_ids"])
    first_e02 = by_id[result.answers["evidence"]["E02"]["record_ids"][0]].timestamp
    assert last_e01 < first_e02


def test_outputs_are_written_and_verifiable(scenario_path: Path, pool: FewShotPool, tmp_path: Path) -> None:
    """저장한 레코드는 무결성 검증을 통과하고 검수용 대화록에 증거 표시가 있다."""
    from evidence_trace.ingest.records import EvidenceRecord

    result = generate(load_scenario(scenario_path), FakeModel(GOOD), pool)
    out = tmp_path / "generated"
    write_outputs(result, out)
    for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines():
        EvidenceRecord.from_dict(json.loads(line))
    review = (out / "review.md").read_text(encoding="utf-8")
    assert "**[E01]**" in review and "(오답 후보)" in review


# ---------- 검사기 ----------

BEAT = Beat(0, "say", datetime(2026, 3, 3, 19, 20, tzinfo=KST), ("suspect",), "선입금 요구", (1, 3),
            evidence_id="E01", markers=("#@금융#",))


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ([{"speaker": "A", "text": "입금 먼저요 #@금융#"}], []),
        ([{"speaker": "A", "text": "입금 먼저요"}], ["필수 표시 #@금융# 누락"]),
        ([{"speaker": "B", "text": "#@금융#"}], ["허용되지 않은 발화자"]),
        ([{"speaker": "A", "text": "#@금융# " + "가" * 40}], ["자"]),
        ([{"speaker": "A", "text": "#@금융# 😀"}, {"speaker": "A", "text": "ㅋㅋ 😀"}], ["이모지"]),
        ([{"speaker": "A", "text": "#@금융# #@비밀#"}], ["알 수 없는 표시"]),
        ([{"speaker": "A", "text": "#@금융# #@이모티콘#흑흑#"}], []),
        ([{"speaker": "A", "text": "실제 대화 예시 문장"}, {"speaker": "A", "text": "#@금융#"}], ["베낌"]),
        ([{"speaker": "A", "text": "#@금융#"}] + [{"speaker": "A", "text": f"네 {i}"} for i in range(3)],
         ["메시지 수"]),
    ],
)
def test_validate_messages(messages: list[dict[str, str]], expected: list[str]) -> None:
    """검사기가 규칙 위반을 사유와 함께 찾아낸다."""
    problems = validate_messages(messages, BEAT, {"A"}, {"실제 대화 예시 문장"})
    if not expected:
        assert problems == []
    else:
        assert len(problems) == len(expected)
        for problem, keyword in zip(problems, expected):
            assert keyword in problem


# ---------- 시나리오 검사 ----------

@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("evidence_id: E02", "evidence_id: E01", "중복된 증거 ID"),
        ('at: "2026-03-03T19:30"', 'at: "2026-03-03T19:00"', "시각이 빠르거나"),
        ("speaker: victim,", "speaker: friend,", "참여자가 아닌 발화자"),
        ('at: "2026-03-07T22:10"', 'at: "2026-04-07T22:10"', "사건 기간 밖"),
    ],
)
def test_invalid_scenarios_are_rejected(tmp_path: Path, old: str, new: str, message: str) -> None:
    """규칙을 어긴 시나리오는 위반 위치와 함께 거부된다."""
    assert old in SCENARIO_YAML
    path = tmp_path / "bad.yaml"
    path.write_text(SCENARIO_YAML.replace(old, new), encoding="utf-8")
    with pytest.raises(ScenarioError, match=message):
        load_scenario(path)


def test_owner_is_inferred_from_empty_contact(scenario_path: Path) -> None:
    """owner 키가 없으면 연락처 이름이 빈 인물을 소유자로 본다."""
    assert load_scenario(scenario_path).owner == "suspect"


def test_schedule_times_rejects_overlap() -> None:
    """다음 장면까지 1분도 안 남으면 배치를 거부한다."""
    start = datetime(2026, 3, 3, 19, 20, tzinfo=KST)
    with pytest.raises(ScenarioError):
        schedule_times(start, 3, start, random.Random(0))


def test_real_scenario_file_is_valid() -> None:
    """레포의 실제 시나리오 파일이 모든 규칙을 통과한다."""
    path = Path(__file__).parents[1] / "data/scenarios/fraud_case_01/scenario.yaml"
    if not path.exists():
        pytest.skip("시나리오 파일 없음")
    scenario = load_scenario(path)
    assert scenario.owner == "suspect"
    assert sum(1 for t in scenario.threads for b in t.beats if b.evidence_id) == 19


# ---------- 후처리 ----------

def test_split_long_text_breaks_at_punctuation() -> None:
    """40자를 넘는 메시지는 문장부호 위치에서 40자 이하 조각으로 나뉜다."""
    text = "안녕하세요! 중고거래 앱에서 판매글 보고 연락드렸어요. 혹시 아직 판매 중이신가요? 상태가 궁금해서요"
    pieces = split_long_text(text)
    assert len(pieces) > 1
    assert all(len(p) <= 40 for p in pieces)
    assert " ".join(pieces) == text


def test_split_keeps_unbreakable_text_for_validator() -> None:
    """쪼갤 경계가 없는 긴 메시지는 그대로 두어 검사기가 반려하게 한다."""
    text = "가" * 50
    assert split_long_text(text) == [text]


def test_normalize_removes_emoji_and_drops_empty() -> None:
    """이모지를 지우고, 이모지만 있던 메시지는 버린다."""
    out, stats = normalize_messages([
        {"speaker": "A", "text": "네 좋아요 👍️"},
        {"speaker": "B", "text": "😊"},
    ])
    assert out == [{"speaker": "A", "text": "네 좋아요"}]
    assert stats == {"split": 0, "emoji_removed": 2}


def test_long_emoji_output_is_fixed_not_rejected(scenario_path: Path, pool: FewShotPool) -> None:
    """EXAONE처럼 긴 존댓말에 이모지를 붙인 출력도 후처리로 합격한다."""
    long_reply = _reply(
        ("구매자", "안녕하세요! 중고거래 앱에서 판매글 보고 연락드렸어요. 혹시 아직 판매 중이신가요? 😊"),
        ("판매자", "네 아직 있어요! 상태 아주 좋고 구성품도 전부 다 있습니다. 🙂"),
    )
    model = FakeModel([long_reply, GOOD[1], GOOD[2]])
    result = generate(load_scenario(scenario_path), model, pool)
    first = result.log[0]
    assert first["ok"] and first["split"] >= 1 and first["emoji_removed"] == 2
    assert all(len(r.content) <= 40 for r in result.records)


def test_count_is_checked_on_raw_llm_output(scenario_path: Path, pool: FewShotPool) -> None:
    """메시지 수는 쪼개기 전 LLM 원본으로 판단한다."""
    too_many = _reply(*[("판매자", "네") for _ in range(4)])  # 첫 장면 허용 2~3개
    model = FakeModel([too_many, GOOD[0], GOOD[1], GOOD[2]])
    result = generate(load_scenario(scenario_path), model, pool)
    assert "메시지 수 4개" in result.log[0]["problems"][0]


def test_progress_is_reported(scenario_path: Path, pool: FewShotPool) -> None:
    """장면마다 진행 상황이 보고된다."""
    lines: list[str] = []
    generate(load_scenario(scenario_path), FakeModel(GOOD), pool, progress=lines.append)
    assert lines[0].startswith("[1/5] t_victim 장면 0 시도 1: 합격")
    assert any("고정 문구" in line for line in lines)


# ---------- event 장면 ----------

def test_event_is_inserted_by_code_but_not_evidence(scenario_path: Path, pool: FewShotPool) -> None:
    """event 장면은 코드가 넣고, 정답지의 증거에는 들어가지 않고 오답 후보에 들어간다."""
    result = generate(load_scenario(scenario_path), FakeModel(GOOD), pool)
    friend_ids = result.answers["decoys"]["t_friend"]
    by_id = {r.record_id: r for r in result.records}
    assert [by_id[i].content for i in friend_ids][-1] == "#@시스템#송금#"
    evidence_ids = {i for e in result.answers["evidence"].values() for i in e["record_ids"]}
    assert not evidence_ids & set(friend_ids)


def test_chat_intent_with_system_marker_is_rejected(tmp_path: Path) -> None:
    """chat 흐름 설명에 시스템 표시를 쓰면 모순된 지시라 시나리오를 거부한다."""
    path = tmp_path / "bad.yaml"
    path.write_text(SCENARIO_YAML.replace("intent: 더치페이", "intent: 더치페이 (#@시스템#송금# 사용)"),
                    encoding="utf-8")
    with pytest.raises(ScenarioError, match="event 장면으로 분리"):
        load_scenario(path)


def test_event_requires_literal(tmp_path: Path) -> None:
    """event 장면에 literal이 없으면 거부한다."""
    path = tmp_path / "bad.yaml"
    path.write_text(SCENARIO_YAML.replace(', literal: "#@시스템#송금#"}\n"""', '}\n"""').replace(
        'speaker: suspect, literal: "#@시스템#송금#"}', "speaker: suspect}"), encoding="utf-8")
    with pytest.raises(ScenarioError, match="literal이 필요"):
        load_scenario(path)


def test_other_real_markers_are_allowed() -> None:
    """AI Hub 원본에 있는 #@번호# 같은 표시는 허용한다."""
    beat = Beat(0, "chat", datetime(2026, 3, 3, 10, 0, tzinfo=KST), ("a",), "송장", (1, 2))
    assert validate_messages([{"speaker": "나", "text": "송장 #@번호# 이에요"}], beat, {"나"}, set()) == []


# ---------- 역할 이름과 강화된 검사 ----------

def test_speaker_is_constrained_to_role_labels(scenario_path: Path, pool: FewShotPool) -> None:
    """LLM 응답 스키마의 발화자가 그 장면의 역할 이름으로 제한된다."""
    model = FakeModel(GOOD)
    generate(load_scenario(scenario_path), model, pool)
    say_schema = model.schemas[1]
    speaker = say_schema["properties"]["messages"]["items"]["properties"]["speaker"]
    assert speaker["enum"] == ["판매자"]
    assert "- 판매자: 사기 판매자" in model.calls[0]


def test_decoy_prompt_hides_case_summary(scenario_path: Path, pool: FewShotPool) -> None:
    """오답 후보 대화방 프롬프트에는 사건 개요가 들어가지 않는다."""
    model = FakeModel(GOOD)
    generate(load_scenario(scenario_path), model, pool)
    assert "테스트용 사건." in model.calls[0]
    assert "테스트용 사건." not in model.calls[2]


def test_roles_must_cover_all_participants(tmp_path: Path) -> None:
    """roles가 참여자를 빠뜨리면 거부한다."""
    path = tmp_path / "bad.yaml"
    path.write_text(SCENARIO_YAML.replace("      victim: {label: 구매자, desc: 구매자}\n", ""),
                    encoding="utf-8")
    with pytest.raises(ScenarioError, match="roles에 참여자 victim"):
        load_scenario(path)


def test_role_labels_must_be_unique(tmp_path: Path) -> None:
    """한 대화방에서 역할 이름이 겹치면 거부한다."""
    path = tmp_path / "bad.yaml"
    path.write_text(SCENARIO_YAML.replace("{label: 구매자, desc: 구매자}", "{label: 판매자, desc: 구매자}"),
                    encoding="utf-8")
    with pytest.raises(ScenarioError, match="역할 이름이 겹칩니다"):
        load_scenario(path)


def test_forbidden_names_exclude_role_labels(scenario_path: Path) -> None:
    """연락처·문서용 이름은 금지하되 역할 이름과 겹치는 호칭은 빼준다."""
    assert load_scenario(scenario_path).forbidden_names() == frozenset({"홍길동", "성춘향", "흥부"})


AVOID_BEAT = Beat(0, "say", datetime(2026, 3, 1, 21, 52, tzinfo=KST), ("suspect",), "돌려 말하기", (1, 3),
                  evidence_id="E01", avoid=("계좌", "통장"))


@pytest.mark.parametrize(
    ("text", "keyword"),
    [
        ("잠깐 계좌 좀 빌려줘", "금지어 '계좌'"),
        ("#전화번호#로 연락해", "형식이 잘못된 '#'"),
        ("#내일 오후 3시?", "형식이 잘못된 '#'"),
        ("hurry up 빨리요", "영어 단어"),
        ("₩150만원이에요", "통화 기호"),
        ("성춘향님 감사해요", "이름 노출"),
    ],
)
def test_validator_catches_review_findings(text: str, keyword: str) -> None:
    """1차 검수에서 사람이 찾은 문제를 이제 검사기가 잡는다."""
    problems = validate_messages([{"speaker": "판매자", "text": text}], AVOID_BEAT, {"판매자"}, set(),
                                 forbidden_names=frozenset({"성춘향"}))
    assert any(keyword in p for p in problems), problems


def test_marker_repeated_in_beat_is_rejected() -> None:
    """계좌 표시를 한 장면에서 여러 번 보내면 반려한다."""
    beat = Beat(0, "say", datetime(2026, 3, 11, 20, 30, tzinfo=KST), ("suspect",), "입금 요청", (1, 3),
                evidence_id="E12", markers=("#@금융#",))
    messages = [{"speaker": "판매자", "text": "#@금융# 여기로요"}, {"speaker": "판매자", "text": "#@금융# 확인요"}]
    assert any("2회 반복" in p for p in validate_messages(messages, beat, {"판매자"}, set()))


def test_valid_markers_are_not_flagged_as_stray() -> None:
    """올바른 표시와 URL 표시는 '#' 오류나 영어 단어로 오인하지 않는다."""
    beat = Beat(0, "chat", datetime(2026, 3, 3, 10, 0, tzinfo=KST), ("a",), "송장", (1, 2))
    text = "#@URL# 여기 #@이모티콘#흑흑# PS5 ok"
    assert validate_messages([{"speaker": "나", "text": text}], beat, {"나"}, set()) == []
