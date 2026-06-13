"""
tools/extract_pdf_to_txt.py
----------------------------
knowledge_base/ 폴더 내 PDF를 TXT로 변환하는 유틸리티.

변환된 TXT 파일은 RAG 지식베이스(ChromaDB)에 자동으로 로드됩니다.

[변환 방법 우선순위]
1. Microsoft Word COM 자동화 (Windows, Word 설치 필요)
2. PowerShell + .NET PDF 렌더링
3. PyPDF2 layout 모드
4. 수동 변환 안내 출력

[KoFIU 연차보고서 2024 PDF 특수 상황]
해당 PDF는 모든 텍스트가 벡터 아웃라인(vector outline)으로 처리되어 있어
PyPDF2를 포함한 모든 텍스트 추출 라이브러리로는 추출 불가합니다.
Word COM 자동화 또는 Adobe Acrobat/수동 변환이 필요합니다.

[실행]
    cd GNN-project
    python tools/extract_pdf_to_txt.py

또는 특정 파일만 변환:
    python tools/extract_pdf_to_txt.py --file "knowledge_base/KoFIU_연차보고서_2024.pdf"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import zlib
import re
from pathlib import Path


# ======================================================================
# 방법 1: Microsoft Word COM 자동화 (Windows 전용)
# ======================================================================

def extract_via_word_com(pdf_path: Path, txt_path: Path) -> bool:
    """
    Microsoft Word COM 자동화로 PDF → TXT 변환.
    Word가 설치된 Windows 환경에서 가장 높은 품질의 텍스트를 추출합니다.
    """
    try:
        import comtypes.client  # type: ignore
    except ImportError:
        return False

    abs_pdf = str(pdf_path.resolve())
    abs_txt = str(txt_path.resolve())

    try:
        word = comtypes.client.CreateObject("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(abs_pdf)
        # wdFormatText = 2
        doc.SaveAs(abs_txt, FileFormat=2)
        doc.Close(False)
        word.Quit()
        return txt_path.exists() and txt_path.stat().st_size > 100
    except Exception as e:
        print(f"    Word COM 오류: {e}")
        return False


# ======================================================================
# 방법 2: PowerShell + Word 자동화 스크립트
# ======================================================================

def extract_via_powershell_word(pdf_path: Path, txt_path: Path) -> bool:
    """
    PowerShell을 통해 Word를 자동화하여 PDF → TXT 변환.
    comtypes 없이 Windows PowerShell만 있으면 동작합니다.
    """
    if os.name != "nt":
        return False  # Windows 전용

    ps_script = f"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {{
    $doc = $word.Documents.Open('{str(pdf_path.resolve()).replace(chr(39), "''")}')
    $doc.SaveAs2('{str(txt_path.resolve()).replace(chr(39), "''")}', 2)
    $doc.Close($false)
    Write-Output 'SUCCESS'
}} catch {{
    Write-Error $_.Exception.Message
}} finally {{
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}}
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if "SUCCESS" in result.stdout and txt_path.exists():
            return txt_path.stat().st_size > 100
        if result.stderr:
            print(f"    PowerShell 오류: {result.stderr.strip()[:200]}")
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    PowerShell 실행 실패: {e}")
        return False


# ======================================================================
# 방법 3: PyPDF2 텍스트 추출 (layout 모드 포함)
# ======================================================================

def extract_via_pypdf2(pdf_path: Path) -> str:
    """PyPDF2로 텍스트 추출 (layout 모드 시도 → 일반 모드 폴백)."""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        return ""

    try:
        reader = PdfReader(str(pdf_path))
        pages: list[str] = []
        for page in reader.pages:
            try:
                # PyPDF2 3.1+ layout 모드 시도
                try:
                    t = page.extract_text(extraction_mode="layout") or ""
                except TypeError:
                    t = page.extract_text() or ""
                pages.append(t)
            except Exception:
                pass
        return "\n".join(pages).strip()
    except Exception as e:
        print(f"    PyPDF2 오류: {e}")
        return ""


# ======================================================================
# 방법 4: 저수준 스트림 파서 (단순 구조 PDF용 폴백)
# ======================================================================

def extract_via_stream_parser(pdf_path: Path) -> str:
    """
    zlib 압축 스트림에서 hex 인코딩된 한글 텍스트를 직접 추출.
    ToUnicode CMap이 있는 PDF에만 유효합니다.
    """
    try:
        with open(pdf_path, "rb") as f:
            raw = f.read()
    except Exception:
        return ""

    stream_pat = re.compile(rb"stream\r?\n(.*?)endstream", re.DOTALL)
    bt_et_pat  = re.compile(rb"BT(.*?)ET", re.DOTALL)
    hex_tj_pat = re.compile(rb"<([0-9A-Fa-f]{2,})>\s*Tj")
    arr_tj_pat = re.compile(rb"\[([^\]]*)\]\s*TJ")
    lit_tj_pat = re.compile(rb"\(([^\)]+)\)\s*Tj")

    text_chunks: list[str] = []

    def _try_decode(raw_bytes: bytes) -> str:
        for enc in ("utf-16-be", "utf-8", "cp949", "euc-kr"):
            try:
                s = raw_bytes.decode(enc, errors="strict")
                if any("\uAC00" <= c <= "\uD7A3" for c in s):
                    return s
            except (UnicodeDecodeError, LookupError):
                pass
        return ""

    for sm in stream_pat.finditer(raw):
        try:
            dec = zlib.decompress(sm.group(1))
        except Exception:
            continue
        for bm in bt_et_pat.finditer(dec):
            block = bm.group(1)
            # <hex> Tj
            for hm in hex_tj_pat.finditer(block):
                decoded = _try_decode(bytes.fromhex(hm.group(1).decode()))
                if decoded:
                    text_chunks.append(decoded)
            # [<hex>...] TJ
            for am in arr_tj_pat.finditer(block):
                parts = re.findall(rb"<([0-9A-Fa-f]+)>", am.group(1))
                chunk = "".join(
                    _try_decode(bytes.fromhex(p.decode())) for p in parts
                )
                if chunk:
                    text_chunks.append(chunk)
            # (literal) Tj
            for lm in lit_tj_pat.finditer(block):
                decoded = _try_decode(lm.group(1))
                if decoded:
                    text_chunks.append(decoded)

    return "\n".join(text_chunks)


# ======================================================================
# 메인 변환 로직
# ======================================================================

def convert_pdf_to_txt(pdf_path: Path, overwrite: bool = False) -> bool:
    """
    단일 PDF 파일을 TXT로 변환합니다.

    Args:
        pdf_path  (Path): 변환할 PDF 파일 경로
        overwrite (bool): True면 기존 TXT 덮어쓰기

    Returns:
        bool: 성공 여부
    """
    txt_path = pdf_path.with_suffix(".txt")

    if txt_path.exists() and not overwrite:
        size_kb = txt_path.stat().st_size // 1024
        print(f"  [SKIP] 건너뜀 (이미 존재, {size_kb}KB): {txt_path.name}")
        return True

    print(f"\n[변환 중] {pdf_path.name}")

    # ── 방법 1: Word COM ───────────────────────────────────────────────
    print("  [1/4] Word COM 자동화 시도...")
    if extract_via_word_com(pdf_path, txt_path):
        size_kb = txt_path.stat().st_size // 1024
        print(f"  [OK] Word COM 성공 ({size_kb}KB) → {txt_path.name}")
        return True

    # ── 방법 2: PowerShell + Word ─────────────────────────────────────
    print("  [2/4] PowerShell + Word 자동화 시도...")
    if extract_via_powershell_word(pdf_path, txt_path):
        size_kb = txt_path.stat().st_size // 1024
        print(f"  [OK] PowerShell+Word 성공 ({size_kb}KB) → {txt_path.name}")
        return True

    # ── 방법 3: PyPDF2 ────────────────────────────────────────────────
    print("  [3/4] PyPDF2 추출 시도...")
    text = extract_via_pypdf2(pdf_path)
    if text and len(text) > 200:
        txt_path.write_text(text, encoding="utf-8")
        print(f"  [OK] PyPDF2 성공 ({len(text):,}자) → {txt_path.name}")
        return True
    print(f"    결과: {len(text)}자 (부족)")

    # ── 방법 4: 스트림 파서 ───────────────────────────────────────────
    print("  [4/4] 저수준 스트림 파서 시도...")
    text = extract_via_stream_parser(pdf_path)
    if text and len(text) > 200:
        txt_path.write_text(text, encoding="utf-8")
        print(f"  [OK] 스트림 파서 성공 ({len(text):,}자) → {txt_path.name}")
        return True
    print(f"    결과: {len(text)}자 (부족)")

    # ── 모두 실패: 수동 안내 ──────────────────────────────────────────
    print(f"""
  [FAIL] 자동 변환 실패: {pdf_path.name}
     이 PDF는 텍스트가 벡터 아웃라인으로 처리된 고급 디자인 PDF입니다.
     아래 방법 중 하나로 수동 변환 후 '{txt_path}' 에 저장하세요.

     ┌─ 방법 A (Microsoft Word) ─────────────────────────────────────┐
     │ 1. Word에서 PDF 열기 (파일 → 열기 → PDF 선택)                │
     │ 2. 편집 허용 클릭                                             │
     │ 3. 다른 이름으로 저장 → 파일 형식: 일반 텍스트(.txt)         │
     └───────────────────────────────────────────────────────────────┘

     ┌─ 방법 B (Adobe Acrobat Reader) ───────────────────────────────┐
     │ 1. PDF 열기 → 편집 → 모두 선택 → 복사                       │
     │ 2. 메모장에 붙여넣기 → '{txt_path.name}' 로 저장             │
     └───────────────────────────────────────────────────────────────┘

     ┌─ 방법 C (온라인 변환) ────────────────────────────────────────┐
     │ smallpdf.com, ilovepdf.com, pdf2doc.com 등에서 PDF→TXT 변환  │
     │ 결과 파일을 '{txt_path}' 로 저장하세요.                      │
     └───────────────────────────────────────────────────────────────┘
""")
    return False


# ======================================================================
# CLI 진입점
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="knowledge_base/ PDF → TXT 변환 (RAG 지식베이스용)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="변환할 PDF 경로 (생략 시 knowledge_base/ 전체 처리)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 TXT 파일 덮어쓰기",
    )
    args = parser.parse_args()

    if args.file:
        pdf_files = [Path(args.file)]
        if not pdf_files[0].exists():
            print(f"오류: '{args.file}' 파일을 찾을 수 없습니다.")
            sys.exit(1)
    else:
        kb_dir = Path("knowledge_base")
        if not kb_dir.exists():
            print("오류: 'knowledge_base/' 폴더가 없습니다. 프로젝트 루트에서 실행하세요.")
            sys.exit(1)
        pdf_files = sorted(kb_dir.glob("*.pdf"))
        if not pdf_files:
            print("knowledge_base/ 에 PDF 파일이 없습니다.")
            sys.exit(0)

    print(f"=== PDF → TXT 변환 시작 ({len(pdf_files)}개 파일) ===")
    ok = sum(convert_pdf_to_txt(p, overwrite=args.overwrite) for p in pdf_files)

    print(f"\n{'='*50}")
    print(f"완료: {ok}/{len(pdf_files)}개 변환 성공")
    if ok == len(pdf_files):
        print("\n[OK] 모든 파일 준비 완료!")
        print("   앱을 재시작하면 ChromaDB에 자동 로드됩니다.")
        print("   (기존 chroma_db/ 폴더를 삭제 후 재시작하면 재빌드됩니다)")
    else:
        print("\n[WARN] 일부 파일은 수동 변환이 필요합니다.")
        print("   변환 후 'python tools/extract_pdf_to_txt.py' 를 다시 실행하세요.")


if __name__ == "__main__":
    main()
