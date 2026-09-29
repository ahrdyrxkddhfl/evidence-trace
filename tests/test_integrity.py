"""무결성 해시 불변조건 테스트.

증거 레코드는 저장 후 다시 읽을 때 한 글자라도 바뀌면 반드시 드러나야 한다.
"""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from evidence_trace.ingest.records import (
    Direction,
    EvidenceRecord,
    IntegrityError,
    MessageKind,
    SourceType,
)

KST = timezone(timedelta(hours=9))


@pytest.fixture
def record() -> EvidenceRecord:
    """테스트용 기본 레코드.

    Returns:
        송금 메시지 한 건을 담은 레코드.
    """
    return EvidenceRecord(
        record_id="aihub:d1:U3",
        source_type=SourceType.MESSENGER,
        app="aihub_sns",
        thread_id="aihub:d1",
        timestamp=datetime(2017, 11, 11, 14, 25, tzinfo=KST),
        sender="aihub:d1:P01",
        content="#@시스템#송금#",
        source_ref={"dataset": "aihub_sns", "utterance_id": "U3"},
        kind=MessageKind.TRANSFER,
        recipients=("aihub:d1:P02",),
    )


def test_roundtrip_preserves_record_and_hash(record: EvidenceRecord) -> None:
    """직렬화 후 복원해도 레코드와 해시가 그대로다."""
    restored = EvidenceRecord.from_dict(record.to_dict())
    assert restored == record
    assert restored.sha256 == record.sha256


def test_tampered_content_is_detected(record: EvidenceRecord) -> None:
    """저장된 본문이 바뀌면 복원 시 IntegrityError가 난다."""
    data = record.to_dict()
    data["content"] = "송금 안 했음"
    with pytest.raises(IntegrityError):
        EvidenceRecord.from_dict(data)


def test_tampered_timestamp_is_detected(record: EvidenceRecord) -> None:
    """저장된 시각이 바뀌면 복원 시 IntegrityError가 난다."""
    data = record.to_dict()
    data["timestamp"] = "2017-11-12T14:25:00+09:00"
    with pytest.raises(IntegrityError):
        EvidenceRecord.from_dict(data)


def test_hash_ignores_derived_kind(record: EvidenceRecord) -> None:
    """본문에서 파생되는 kind는 해시에 영향을 주지 않는다."""
    assert replace(record, kind=MessageKind.TEXT).sha256 == record.sha256


def test_replace_recomputes_hash(record: EvidenceRecord) -> None:
    """증거 필드를 바꿔 새 레코드를 만들면 해시가 새로 계산된다."""
    changed = replace(record, direction=Direction.OUTGOING)
    assert changed.sha256 != record.sha256
    assert changed.sha256 == changed.compute_hash()


def test_naive_timestamp_is_rejected(record: EvidenceRecord) -> None:
    """시간대 없는 시각은 레코드로 만들 수 없다."""
    with pytest.raises(ValueError):
        replace(record, timestamp=datetime(2017, 11, 11, 14, 25))


@pytest.mark.parametrize("field_name", ["record_id", "thread_id", "sender"])
def test_empty_identifier_is_rejected(record: EvidenceRecord, field_name: str) -> None:
    """필수 식별자가 비어 있으면 레코드로 만들 수 없다."""
    with pytest.raises(ValueError):
        replace(record, **{field_name: ""})
