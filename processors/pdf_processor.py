from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from pypdf import PdfReader

from .config_processor import company_key


def read_pdf(path: Path, max_chars: int = 18000) -> str:
    try:
        reader = PdfReader(str(path))
        parts: list[str] = []
        priority = list(reader.pages[:12]) + list(reader.pages[12:40])
        for page in priority:
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts)[:max_chars]
    except Exception:
        return ""


def find_pdf(ipo_dir: Path, company: str) -> Optional[Path]:
    target = company_key(company)
    best: Optional[tuple[int, Path]] = None
    for file_path in ipo_dir.glob("*.pdf"):
        stem = re.sub(r"_DRHP$", "", file_path.stem, flags=re.IGNORECASE).replace("_", " ")
        candidate = company_key(stem)
        if candidate == target:
            return file_path

        target_tokens = set(target.split())
        candidate_tokens = set(candidate.split())
        overlap = len(target_tokens & candidate_tokens)
        if overlap >= max(2, min(len(target_tokens), 3)):
            score = overlap * 100 - abs(len(candidate_tokens) - len(target_tokens))
            if best is None or score > best[0]:
                best = (score, file_path)
    return best[1] if best else None


def list_companies(ipo_dir: Path) -> list[str]:
    names: list[str] = []
    for file_path in sorted(ipo_dir.glob("*.pdf")):
        stem = re.sub(r"_DRHP$", "", file_path.stem, flags=re.IGNORECASE)
        name = stem.replace("_", " ").strip().title()
        if name:
            names.append(name)
    return sorted(set(names))
