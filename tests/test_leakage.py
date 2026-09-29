"""누출 측정 도구 테스트."""

import json
from pathlib import Path

import numpy as np
import pytest

from evidence_trace.eval.leakage import (
    DeviceData,
    discriminator_scores,
    main,
    measure,
    rank_threads,
    style_stats,
    thread_structure,
)
from evidence_trace.ingest.device import write_device
from test_device import aihub_root, device, scenario_file  # noqa: F401  (pytest 픽스처 재사용)


def _message(record_id: str, thread: str, day: str, content: str = "ㅋㅋ") -> dict:
    """측정에 필요한 필드만 가진 기기 메시지."""
    return {"record_id": record_id, "thread_id": thread, "timestamp": f"{day}T21:00:00+09:00",
            "content": content}


def test_style_stats_counts_jamo_and_punctuation() -> None:
    """자모 수, 자모를 쓴 메시지 비율, 문장부호로 끝나는 비율을 센다."""
    stats = style_stats(["ㅋㅋㅋ 뭐해", "밥 먹었어요.", "#@이름# 왔어?"])
    assert stats["jamo_per_message"] == 1.0
    assert stats["jamo_message_ratio"] == round(1 / 3, 4)
    assert stats["ends_with_punct_ratio"] == round(2 / 3, 4)
    assert stats["marker_message_ratio"] == round(1 / 3, 4)


def test_thread_structure_by_origin() -> None:
    """출처별 여러 날 대화방 비율을 계산한다."""
    data = DeviceData(
        messages=[_message("m1", "a", "2026-03-01"), _message("m2", "a", "2026-03-02"),
                  _message("m3", "b", "2026-03-01"), _message("m4", "c", "2026-03-05")],
        origin={"m1": "synthetic", "m2": "synthetic", "m3": "aihub_sns", "m4": "aihub_sns"},
        evidence_threads={"a"}, decoy_threads=set(),
    )
    summary = thread_structure(data)
    assert summary["synthetic"]["multi_day_ratio"] == 1.0
    assert summary["aihub_sns"]["multi_day_ratio"] == 0.0


def test_rank_threads_orders_by_mean_score() -> None:
    """대화방 평균 점수가 높을수록 앞 순위이고, 사건·오답 순위를 따로 보고한다."""
    data = DeviceData(
        messages=[_message("m1", "evid", "2026-03-01"), _message("m2", "decoy", "2026-03-01"),
                  _message("m3", "bg1", "2026-03-01"), _message("m4", "bg2", "2026-03-01")],
        origin={}, evidence_threads={"evid"}, decoy_threads={"decoy"},
    )
    result = rank_threads(data, np.array([0.6, 0.9, 0.1, 0.2]))
    assert result["decoy_ranks"] == [1] and result["evidence_ranks"] == [2]
    assert result["hits"]["top10"] == {"evidence_found": 1, "random_expected": 1.0}


def test_discriminator_requires_enough_messages() -> None:
    """합성 메시지가 폴드 수보다 적으면 거부한다."""
    with pytest.raises(ValueError, match="교차 검증"):
        discriminator_scores(["a", "b", "c", "d", "e", "f"], [1, 0, 0, 0, 0, 0], folds=5)


def test_discriminator_separates_obvious_styles() -> None:
    """문체가 확연히 다르면 판별기가 구분해낸다 (도구가 누출을 잡을 수 있는지 확인)."""
    real = [f"ㅋㅋㅋ 오늘 {i}시에 봐 ㅎㅎ" for i in range(40)]
    synthetic = [f"안녕하세요. {i}번 문의 드립니다." for i in range(40)]
    scores = discriminator_scores(real + synthetic, [0] * 40 + [1] * 40)
    assert scores[40:].mean() > 0.8 and scores[:40].mean() < 0.2


def test_measure_on_assembled_device(device, tmp_path: Path) -> None:  # noqa: F811
    """조립된 기기에서 전체 측정이 돌아가고 사건·오답 대화방이 순위에 잡힌다."""
    _, assembled = device
    write_device(assembled, tmp_path / "dev")
    report = measure(DeviceData.load(tmp_path / "dev"))
    assert 0.0 <= report["discriminator_auc"] <= 1.0
    ranking = report["ranking_by_synthetic_score"]
    assert len(ranking["evidence_ranks"]) == 1 and len(ranking["decoy_ranks"]) == 1
    assert set(report["style"]) == {"background", "synthetic"}


def test_cli_writes_report(device, tmp_path: Path) -> None:  # noqa: F811
    """CLI가 private/leakage_report.json을 저장한다."""
    _, assembled = device
    write_device(assembled, tmp_path / "dev")
    assert main([str(tmp_path / "dev")]) == 0
    report = json.loads((tmp_path / "dev" / "private" / "leakage_report.json").read_text(encoding="utf-8"))
    assert "discriminator_auc" in report
