"""한글 파일명이 들어 있는 zip 아카이브를 안전하게 해제한다.

국내 기관이 배포하는 zip은 파일명 인코딩 표시(UTF-8 플래그)가 빠진 채
UTF-8 또는 CP949로 이름을 저장하는 경우가 많다. macOS 기본 unzip은 이런
이름을 CP437로 해석해 파일명이 깨진다. 이 스크립트는 원래 바이트를 복원한 뒤
UTF-8, CP949 순으로 디코딩을 시도해 올바른 한글 파일명으로 해제한다.

추가로 다음을 보장한다.

* 해제된 파일명은 NFC로 정규화한다. macOS에서 만든 NFD 이름과 섞여
  문자열 비교가 실패하는 문제를 막기 위함이다.
* 아카이브 안의 경로가 출력 폴더 밖을 가리키면(zip slip) 해제를 거부한다.
* 원본 아카이브는 읽기만 하며 수정하지 않는다.

Example:
    프로젝트 루트에서 실행한다::

        $ python3 scripts/extract_archive.py \\
            "data/raw/aihub_sns/original/.../[라벨]한국어SNS_valid.zip" \\
            data/raw/aihub_sns/extracted/valid
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import unicodedata
import zipfile
from pathlib import Path

_UTF8_FLAG = 0x800
"""int: zip 일반 목적 비트 플래그 중 '파일명이 UTF-8'임을 뜻하는 비트."""

_CHUNK_SIZE = 1024 * 1024
"""int: 해시 계산과 파일 복사에 쓰는 읽기 단위(바이트)."""


def decode_member_name(info: zipfile.ZipInfo) -> str:
    """zip 멤버의 파일명을 올바른 한글 문자열로 복원한다.

    UTF-8 플래그가 켜져 있으면 zipfile이 이미 올바르게 디코딩했으므로
    그대로 쓴다. 꺼져 있으면 zipfile은 이름을 CP437로 디코딩해 두었으므로,
    CP437로 다시 인코딩해 원래 바이트를 얻은 뒤 UTF-8, CP949 순으로
    디코딩을 시도한다.

    Args:
        info: 파일명을 복원할 zip 멤버의 메타데이터.

    Returns:
        NFC로 정규화된 파일명. 어떤 인코딩으로도 디코딩되지 않으면
        zipfile이 해석한 이름을 그대로 NFC 정규화해 돌려준다.

    Example:
        >>> info = zipfile.ZipInfo("주거.json".encode("cp949").decode("cp437"))
        >>> decode_member_name(info)
        '주거.json'
    """
    if info.flag_bits & _UTF8_FLAG:
        return unicodedata.normalize("NFC", info.filename)

    raw = info.filename.encode("cp437")
    for encoding in ("utf-8", "cp949"):
        try:
            return unicodedata.normalize("NFC", raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    return unicodedata.normalize("NFC", info.filename)


def resolve_safe_target(dest_dir: Path, member_name: str) -> Path:
    """멤버를 저장할 경로를 계산하고 출력 폴더 밖으로 벗어나지 않는지 검사한다.

    Args:
        dest_dir: 해제 결과를 저장할 폴더. 절대 경로로 변환해 비교한다.
        member_name: 복원된 zip 멤버 이름.

    Returns:
        ``dest_dir`` 안쪽에 위치함이 확인된 절대 경로.

    Raises:
        ValueError: 멤버 경로가 ``../`` 등을 이용해 ``dest_dir`` 밖을
            가리키는 경우(zip slip).
    """
    root = dest_dir.resolve()
    target = (root / member_name).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"출력 폴더 밖을 가리키는 경로라 해제를 거부합니다: {member_name}")
    return target


def sha256_of(path: Path) -> str:
    """파일의 SHA-256 해시를 계산한다.

    Args:
        path: 해시를 계산할 파일 경로.

    Returns:
        16진수 문자열로 표현한 SHA-256 해시.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive: Path, dest_dir: Path, dry_run: bool = False) -> list[Path]:
    """zip 아카이브를 한글 파일명을 복원하며 해제한다.

    큰 파일도 메모리에 통째로 올리지 않도록 스트리밍으로 복사한다.

    Args:
        archive: 해제할 zip 파일 경로. 읽기 전용으로만 연다.
        dest_dir: 해제 결과를 저장할 폴더. 없으면 만든다.
        dry_run: True이면 실제로 해제하지 않고 복원된 파일명만 출력한다.

    Returns:
        해제된(``dry_run``이면 해제될) 파일들의 경로 목록. 폴더 항목은
        포함하지 않는다.

    Raises:
        FileNotFoundError: ``archive``가 존재하지 않는 경우.
        zipfile.BadZipFile: ``archive``가 올바른 zip 파일이 아닌 경우.
        ValueError: 아카이브 안에 출력 폴더 밖을 가리키는 경로가 있는 경우.
    """
    if not archive.is_file():
        raise FileNotFoundError(f"아카이브를 찾을 수 없습니다: {archive}")

    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = decode_member_name(info)
            target = resolve_safe_target(dest_dir, name)

            if info.is_dir():
                if not dry_run:
                    target.mkdir(parents=True, exist_ok=True)
                continue

            print(f"{info.file_size:>12,d}  {name}")
            extracted.append(target)
            if dry_run:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, _CHUNK_SIZE)
    return extracted


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """명령행 인자를 파싱한다.

    Args:
        argv: 파싱할 인자 목록. None이면 ``sys.argv[1:]``을 쓴다.

    Returns:
        ``archive``, ``dest_dir``, ``dry_run`` 속성을 가진 네임스페이스.
    """
    parser = argparse.ArgumentParser(description="한글 파일명을 복원하며 zip을 해제합니다.")
    parser.add_argument("archive", type=Path, help="해제할 zip 파일 경로")
    parser.add_argument("dest_dir", type=Path, help="해제 결과를 저장할 폴더")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제로 해제하지 않고 복원된 파일명만 확인",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """스크립트 진입점. 원본 해시를 출력한 뒤 아카이브를 해제한다.

    Args:
        argv: 명령행 인자 목록. None이면 ``sys.argv[1:]``을 쓴다.

    Returns:
        프로세스 종료 코드. 성공하면 0, 실패하면 1.
    """
    args = parse_args(argv)
    try:
        print(f"원본 SHA-256: {sha256_of(args.archive)}")
        files = extract_archive(args.archive, args.dest_dir, dry_run=args.dry_run)
    except (FileNotFoundError, zipfile.BadZipFile, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    action = "해제 예정" if args.dry_run else "해제 완료"
    print(f"{action}: 파일 {len(files)}개 → {args.dest_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
