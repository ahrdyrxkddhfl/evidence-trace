"""조립된 가상 기기에서 "합성이라서 찾히는" 누출을 측정한다.

사건 증거는 합성이고 배경은 실제 대화다. 검색 시스템이 내용이 아니라 합성 특유의
흔적(문체, 시간 구조)으로 증거를 찾는다면 평가 점수가 부풀려진다. 이 모듈은 그
위험을 숫자로 잰다. 측정은 ``private/``의 출처 정보를 쓰므로 평가 쪽 도구이며,
에이전트나 검색 색인은 이 결과를 보지 않는다.

측정 항목:

1. **문체 통계**: 메시지 길이, 일반 자모(ㅋ·ㅠ 등) 사용, 문장부호로 끝나는 비율,
   이모지, 비식별화 표시 사용을 배경과 합성으로 나눠 비교한다.
2. **시간 구조**: 여러 날에 걸친 대화방 비율과 대화방당 메시지 수.
3. **판별기**: 글자 n-gram TF-IDF와 로지스틱 회귀로 메시지가 합성인지 가린다.
   교차 검증으로 얻은, 학습에 쓰지 않은 예측(out-of-fold)만 쓴다. ROC AUC가
   0.5면 구분 불가, 1.0이면 완벽히 구분된다.
4. **문체만으로 사건 찾기**: 판별기의 "합성 같음" 점수를 대화방별로 평균 내 줄
   세웠을 때 사건 대화방이 몇 위에 오는지, 상위 k개 안에 몇 개가 드는지 본다.
   무작위로 줄 세웠을 때의 기대값과 함께 보고한다. 판별기가 합성을 잘 구분해도
   사건 대화방이 합성 오답 대화방 사이에 묻히면 이 값은 낮아진다.

Example:
    프로젝트 루트에서 실행한다::

        $ python -m evidence_trace.eval.leakage data/processed/devices/fraud_case_01
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

SYNTHETIC = "synthetic"
"""str: 출처 기록에서 합성 메시지를 뜻하는 값."""

_COMPAT_JAMO = re.compile("[\u3131-\u318E]")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_MARKER = re.compile(r"#@[^#\s]+#(?:[^#\s]+#)?")
_ENDS_WITH_PUNCT = re.compile(r"[.!?]$")


@dataclass
class DeviceData:
    """측정에 필요한 기기 데이터.

    Attributes:
        messages: 기기 레코드 딕셔너리 목록 (``device/records.jsonl``).
        origin: 메시지 ID → 출처(``"aihub_sns"`` 또는 ``"synthetic"``).
        evidence_threads: 사건 증거가 들어 있는 대화방 ID.
        decoy_threads: 합성 오답 후보 대화방 ID.
    """

    messages: list[dict[str, Any]]
    origin: dict[str, str]
    evidence_threads: set[str]
    decoy_threads: set[str]

    @classmethod
    def load(cls, device_dir: Path) -> DeviceData:
        """조립 결과 폴더에서 데이터를 읽는다.

        Args:
            device_dir: ``device/``와 ``private/``를 담은 폴더.

        Returns:
            측정용 데이터.

        Raises:
            FileNotFoundError: 필요한 파일이 없는 경우.
        """
        with (device_dir / "device" / "records.jsonl").open(encoding="utf-8") as handle:
            messages = [json.loads(line) for line in handle]
        origin = {}
        with (device_dir / "private" / "provenance.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                origin[entry["device_record_id"]] = entry["origin"]
        answers = json.loads((device_dir / "private" / "answers.json").read_text(encoding="utf-8"))
        evidence = {e["thread"] for e in answers["evidence"].values()}
        return cls(messages, origin, evidence, set(answers["decoys"]))


def style_stats(texts: list[str]) -> dict[str, float]:
    """메시지 묶음의 문체 통계를 계산한다.

    Args:
        texts: 메시지 본문 목록.

    Returns:
        메시지 수, 평균·중앙값 길이, 메시지당 일반 자모 수, 자모를 쓴 메시지
        비율, 문장부호로 끝나는 비율, 이모지 비율, 비식별화 표시를 쓴 메시지 비율.

    Example:
        >>> style_stats(["ㅋㅋ 뭐해", "밥 먹었어."])["ends_with_punct_ratio"]
        0.5
    """
    if not texts:
        return {"messages": 0}
    lengths = [len(t) for t in texts]
    jamo = [len(_COMPAT_JAMO.findall(t)) for t in texts]
    n = len(texts)
    return {
        "messages": n,
        "mean_length": round(statistics.fmean(lengths), 2),
        "median_length": statistics.median(lengths),
        "jamo_per_message": round(sum(jamo) / n, 3),
        "jamo_message_ratio": round(sum(j > 0 for j in jamo) / n, 4),
        "ends_with_punct_ratio": round(sum(bool(_ENDS_WITH_PUNCT.search(t.strip())) for t in texts) / n, 4),
        "emoji_ratio": round(sum(bool(_EMOJI.search(t)) for t in texts) / n, 4),
        "marker_message_ratio": round(sum(bool(_MARKER.search(t)) for t in texts) / n, 4),
    }


def thread_structure(data: DeviceData) -> dict[str, dict[str, float]]:
    """출처별 대화방 시간 구조를 요약한다.

    Args:
        data: 기기 데이터.

    Returns:
        출처 → ``{"threads", "multi_day_ratio", "median_days", "median_messages"}``.
    """
    days: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    kind: dict[str, str] = {}
    for message in data.messages:
        thread = message["thread_id"]
        days[thread].add(message["timestamp"][:10])
        counts[thread] += 1
        kind[thread] = data.origin[message["record_id"]]
    summary = {}
    for label in sorted(set(kind.values())):
        threads = [t for t in kind if kind[t] == label]
        summary[label] = {
            "threads": len(threads),
            "multi_day_ratio": round(sum(len(days[t]) > 1 for t in threads) / len(threads), 4),
            "median_days": statistics.median(len(days[t]) for t in threads),
            "median_messages": statistics.median(counts[t] for t in threads),
        }
    return summary


def discriminator_scores(texts: list[str], labels: list[int], folds: int = 5, seed: int = 0) -> np.ndarray:
    """교차 검증으로 메시지별 "합성일 확률"을 구한다.

    각 메시지의 점수는 그 메시지를 학습에 쓰지 않은 모델이 매긴 값이다.
    합성 메시지가 매우 적으므로 클래스 가중치를 균형 있게 준다.

    Args:
        texts: 메시지 본문.
        labels: 합성이면 1, 실제면 0.
        folds: 교차 검증 겹 수. 합성 메시지 수보다 클 수 없다.
        seed: 폴드 분할 시드.

    Returns:
        메시지별 합성 확률 배열.

    Raises:
        ValueError: 한쪽 클래스의 메시지 수가 ``folds``보다 적은 경우.
    """
    y = np.asarray(labels)
    if min(int(y.sum()), int(len(y) - y.sum())) < folds:
        raise ValueError(f"클래스별 메시지가 {folds}개 이상이어야 교차 검증할 수 있습니다")
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(model, texts, y, cv=splitter, method="predict_proba")[:, 1]


def rank_threads(data: DeviceData, scores: np.ndarray) -> dict[str, Any]:
    """대화방을 평균 "합성 같음" 점수로 줄 세워 사건 대화방의 순위를 본다.

    Args:
        data: 기기 데이터.
        scores: :func:`discriminator_scores`의 결과 (``data.messages`` 순서).

    Returns:
        사건·오답 대화방의 순위(1위가 가장 합성 같음), 상위 k 안의 사건 대화방 수와
        무작위 기대값, 사건 대화방의 평균 순위 백분위.
    """
    per_thread: dict[str, list[float]] = defaultdict(list)
    for message, score in zip(data.messages, scores):
        per_thread[message["thread_id"]].append(float(score))
    ordered = sorted(per_thread, key=lambda t: (-statistics.fmean(per_thread[t]), t))
    rank = {thread: i for i, thread in enumerate(ordered, start=1)}
    total = len(ordered)
    evidence_ranks = sorted(rank[t] for t in data.evidence_threads if t in rank)
    decoy_ranks = sorted(rank[t] for t in data.decoy_threads if t in rank)
    hits = {}
    for k in (10, 50, 100):
        found = sum(r <= k for r in evidence_ranks)
        expected = len(evidence_ranks) * min(k, total) / total
        hits[f"top{k}"] = {"evidence_found": found, "random_expected": round(expected, 2)}
    percentile = statistics.fmean(r / total for r in evidence_ranks) if evidence_ranks else None
    return {
        "threads": total,
        "evidence_ranks": evidence_ranks,
        "decoy_ranks": decoy_ranks,
        "hits": hits,
        "evidence_mean_rank_percentile": round(percentile, 4) if percentile is not None else None,
    }


def measure(data: DeviceData, folds: int = 5, seed: int = 0) -> dict[str, Any]:
    """누출 측정 전체를 실행한다.

    Args:
        data: 기기 데이터.
        folds: 판별기 교차 검증 겹 수.
        seed: 폴드 분할 시드.

    Returns:
        문체 통계, 시간 구조, 판별기 AUC, 대화방 순위를 담은 보고서.
    """
    texts = [m["content"] for m in data.messages]
    labels = [int(data.origin[m["record_id"]] == SYNTHETIC) for m in data.messages]
    scores = discriminator_scores(texts, labels, folds=folds, seed=seed)
    return {
        "style": {
            "background": style_stats([t for t, y in zip(texts, labels) if y == 0]),
            "synthetic": style_stats([t for t, y in zip(texts, labels) if y == 1]),
        },
        "thread_structure": thread_structure(data),
        "discriminator_auc": round(float(roc_auc_score(labels, scores)), 4),
        "ranking_by_synthetic_score": rank_threads(data, scores),
    }


def _print_report(report: dict[str, Any]) -> None:
    """보고서의 핵심 수치를 읽기 쉽게 출력한다.

    Args:
        report: :func:`measure`의 결과.
    """
    print("[문체]")
    for key in ("mean_length", "jamo_per_message", "jamo_message_ratio", "ends_with_punct_ratio",
                "emoji_ratio", "marker_message_ratio"):
        bg, syn = report["style"]["background"].get(key), report["style"]["synthetic"].get(key)
        print(f"  {key:<24} 배경 {bg:<8} 합성 {syn}")
    print("[시간 구조]")
    for label, values in report["thread_structure"].items():
        print(f"  {label:<10} {values}")
    ranking = report["ranking_by_synthetic_score"]
    print(f"[판별기] AUC {report['discriminator_auc']}")
    print(f"[문체 점수로 줄 세우기] 대화방 {ranking['threads']}개 중")
    print(f"  사건 대화방 순위 {ranking['evidence_ranks']}, 오답 합성 대화방 순위 {ranking['decoy_ranks']}")
    for k, v in ranking["hits"].items():
        print(f"  {k}: 사건 대화방 {v['evidence_found']}개 (무작위 기대 {v['random_expected']})")


def main(argv: list[str] | None = None) -> int:
    """명령행 진입점. 측정하고 ``private/leakage_report.json``에 저장한다.

    Args:
        argv: 명령행 인자 목록. None이면 ``sys.argv[1:]``을 쓴다.

    Returns:
        프로세스 종료 코드. 성공하면 0, 실패하면 1.
    """
    parser = argparse.ArgumentParser(description="가상 기기의 합성 데이터 누출 측정")
    parser.add_argument("device_dir", type=Path, help="조립 결과 폴더 (device/, private/ 포함)")
    parser.add_argument("--folds", type=int, default=5, help="교차 검증 겹 수")
    parser.add_argument("--seed", type=int, default=0, help="폴드 분할 시드")
    args = parser.parse_args(argv)
    try:
        report = measure(DeviceData.load(args.device_dir), folds=args.folds, seed=args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    out = args.device_dir / "private" / "leakage_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(report)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
