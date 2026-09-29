"""모든 증거 출처가 공유하는 공통 레코드 스키마.

메신저, 문자, 통화기록처럼 출처가 달라도 이 모듈의 :class:`EvidenceRecord`
하나로 표현한다. 검색·에이전트·평가 모듈은 출처를 몰라도 이 형식만 다루면 된다.

레코드 하나는 원본 저장소의 행 하나(메시지 한 줄)에 대응한다. 검색할 때는
여러 레코드를 묶은 창(window)을 쓰더라도, 인용은 항상 이 레코드 단위로
되돌아온다.

무결성:
    레코드가 만들어질 때 증거로서 의미가 있는 필드(:data:`INTEGRITY_FIELDS`)의
    정규화된 JSON으로 SHA-256을 계산해 ``sha256``에 저장한다. 저장했다가 다시
    읽을 때 해시가 다르면 :class:`IntegrityError`를 던져 변조를 드러낸다.
    ``kind`` 처럼 본문에서 파생되는 값은 해시에 넣지 않는다.

Example:
    >>> from datetime import datetime, timezone, timedelta
    >>> kst = timezone(timedelta(hours=9))
    >>> record = EvidenceRecord(
    ...     record_id="aihub:d1:U1",
    ...     source_type=SourceType.MESSENGER,
    ...     app="aihub_sns",
    ...     thread_id="aihub:d1",
    ...     timestamp=datetime(2017, 11, 11, 14, 18, tzinfo=kst),
    ...     sender="aihub:d1:P01",
    ...     content="아 헐",
    ...     source_ref={"dataset": "aihub_sns", "utterance_id": "U1"},
    ... )
    >>> EvidenceRecord.from_dict(record.to_dict()) == record
    True
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    """증거가 나온 저장소의 종류.

    Attributes:
        MESSENGER: 메신저 앱 대화.
        SMS: 문자 메시지.
        CALL: 통화 기록.
    """

    MESSENGER = "messenger"
    SMS = "sms"
    CALL = "call"


class Direction(str, Enum):
    """기기 소유자 기준 메시지 방향.

    원본 데이터에 소유자 개념이 없으면(예: AI Hub 원본) ``None``으로 두고,
    가상 기기를 조립하는 단계에서 채운다.

    Attributes:
        INCOMING: 소유자가 받은 메시지.
        OUTGOING: 소유자가 보낸 메시지.
    """

    INCOMING = "incoming"
    OUTGOING = "outgoing"


class MessageKind(str, Enum):
    """본문에서 판별한 메시지 종류.

    본문에서 파생되는 값이므로 무결성 해시에는 포함하지 않는다.

    Attributes:
        TEXT: 일반 텍스트.
        EMOTICON: 이모티콘만으로 이루어진 메시지.
        PHOTO: 사진 전송.
        VIDEO: 동영상 전송.
        FILE: 파일 전송.
        TRANSFER: 송금.
        DELETED: 삭제된 메시지의 흔적.
        MAP: 지도·위치 공유.
        SEARCH: 검색 결과 공유.
        OTHER_SYSTEM: 그 밖의 시스템 메시지.
    """

    TEXT = "text"
    EMOTICON = "emoticon"
    PHOTO = "photo"
    VIDEO = "video"
    FILE = "file"
    TRANSFER = "transfer"
    DELETED = "deleted"
    MAP = "map"
    SEARCH = "search"
    OTHER_SYSTEM = "other_system"


INTEGRITY_FIELDS: tuple[str, ...] = (
    "record_id",
    "source_type",
    "app",
    "thread_id",
    "timestamp",
    "sender",
    "recipients",
    "direction",
    "content",
    "source_ref",
)
"""tuple[str, ...]: 무결성 해시 계산에 쓰는 필드. 순서는 결과에 영향이 없다."""


class IntegrityError(ValueError):
    """저장된 해시와 다시 계산한 해시가 다를 때 발생한다."""


def _canonical_json(payload: dict[str, Any]) -> str:
    """해시 계산용으로 키 순서와 공백을 고정한 JSON 문자열을 만든다.

    Args:
        payload: 직렬화할 딕셔너리. JSON으로 표현 가능한 값만 담겨야 한다.

    Returns:
        키를 정렬하고 공백을 제거한 JSON 문자열. 한글은 이스케이프하지 않는다.
    """
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class EvidenceRecord:
    """증거 레코드 한 건. 원본 저장소의 행 하나에 대응한다.

    불변(frozen) 객체다. 값을 바꿀 때는 :func:`dataclasses.replace`로 새
    레코드를 만들며, 이때 ``sha256``도 새로 계산된다.

    Attributes:
        record_id: 전체 증거 안에서 유일한 레코드 식별자.
            예: ``"aihub:<dialogueID>:U3"``.
        source_type: 증거 저장소 종류.
        app: 증거를 만든 앱 또는 데이터셋 이름. 예: ``"aihub_sns"``.
        thread_id: 레코드가 속한 대화방(스레드) 식별자.
        timestamp: 메시지 시각. 시간대 정보가 반드시 있어야 한다.
        sender: 발신자 식별자. 전체 증거 안에서 유일해야 한다.
        content: 메시지 본문. 원본을 그대로 보존하며 가공하지 않는다.
        source_ref: 원본 위치 정보(데이터셋, 파일, 원본 ID 등).
        kind: 본문에서 판별한 메시지 종류.
        recipients: 수신자 식별자 목록.
        direction: 기기 소유자 기준 방향. 알 수 없으면 ``None``.
        sha256: :data:`INTEGRITY_FIELDS`로 계산한 무결성 해시. 생성 시 자동
            계산되며 직접 지정할 수 없다.

    Raises:
        ValueError: ``record_id``, ``thread_id``, ``sender`` 중 빈 값이 있거나
            ``timestamp``에 시간대 정보가 없는 경우.
    """

    record_id: str
    source_type: SourceType
    app: str
    thread_id: str
    timestamp: datetime
    sender: str
    content: str
    source_ref: dict[str, str]
    kind: MessageKind = MessageKind.TEXT
    recipients: tuple[str, ...] = ()
    direction: Direction | None = None
    sha256: str = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """필수 값을 검증하고 무결성 해시를 계산한다.

        Raises:
            ValueError: 필수 식별자가 비어 있거나 시각에 시간대가 없는 경우.
        """
        for name in ("record_id", "thread_id", "sender"):
            if not getattr(self, name):
                raise ValueError(f"{name}은(는) 비어 있을 수 없습니다")
        if self.timestamp.tzinfo is None:
            raise ValueError(f"timestamp에 시간대 정보가 없습니다: {self.record_id}")
        object.__setattr__(self, "sha256", self.compute_hash())

    def _integrity_payload(self) -> dict[str, Any]:
        """무결성 해시 대상 필드만 JSON 표현 가능한 형태로 모은다.

        Returns:
            :data:`INTEGRITY_FIELDS`를 키로 가지는 딕셔너리.
        """
        full = self._serializable_fields()
        return {name: full[name] for name in INTEGRITY_FIELDS}

    def _serializable_fields(self) -> dict[str, Any]:
        """``sha256``을 제외한 모든 필드를 JSON 표현 가능한 값으로 변환한다.

        Returns:
            열거형은 문자열로, 시각은 ISO 8601 문자열로, 튜플은 리스트로 바꾼
            딕셔너리.
        """
        return {
            "record_id": self.record_id,
            "source_type": self.source_type.value,
            "app": self.app,
            "thread_id": self.thread_id,
            "timestamp": self.timestamp.isoformat(),
            "sender": self.sender,
            "recipients": list(self.recipients),
            "direction": self.direction.value if self.direction else None,
            "content": self.content,
            "source_ref": dict(self.source_ref),
            "kind": self.kind.value,
        }

    def compute_hash(self) -> str:
        """현재 필드 값으로 무결성 해시를 계산한다.

        Returns:
            16진수 SHA-256 문자열.
        """
        encoded = _canonical_json(self._integrity_payload()).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """JSONL 저장용 딕셔너리로 변환한다.

        Returns:
            모든 필드와 ``sha256``을 담은 JSON 표현 가능한 딕셔너리.
        """
        data = self._serializable_fields()
        data["sha256"] = self.sha256
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], verify: bool = True) -> EvidenceRecord:
        """:meth:`to_dict` 결과로부터 레코드를 복원한다.

        Args:
            data: :meth:`to_dict` 형식의 딕셔너리.
            verify: True이면 저장된 ``sha256``과 다시 계산한 해시를 비교한다.

        Returns:
            복원된 레코드.

        Raises:
            KeyError: 필수 키가 없는 경우.
            ValueError: 열거형 값이나 시각 형식이 잘못된 경우.
            IntegrityError: ``verify``가 True이고 해시가 일치하지 않는 경우.
        """
        direction = data.get("direction")
        record = cls(
            record_id=data["record_id"],
            source_type=SourceType(data["source_type"]),
            app=data["app"],
            thread_id=data["thread_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            sender=data["sender"],
            content=data["content"],
            source_ref=dict(data["source_ref"]),
            kind=MessageKind(data.get("kind", MessageKind.TEXT.value)),
            recipients=tuple(data.get("recipients", ())),
            direction=Direction(direction) if direction else None,
        )
        stored = data.get("sha256")
        if verify and stored != record.sha256:
            raise IntegrityError(
                f"무결성 해시 불일치: {record.record_id} (저장값 {stored}, 계산값 {record.sha256})"
            )
        return record
