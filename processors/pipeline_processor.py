from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from .pdf_processor import find_pdf, read_pdf, list_companies


def trigger_extraction(
    cache: dict[str, dict],
    inflight: set[str],
    lock: threading.Lock,
    key_func: Callable[[str], str],
    extractor: Callable[[str, str], dict],
    ipo_dir: Path,
    company: str,
    force: bool = False,
) -> None:
    key = key_func(company)
    with lock:
        if key in inflight:
            return
        if not force and key in cache and cache[key].get("status") == "extracted":
            age = time.time() - cache[key].get("extracted_at", 0)
            if age < 3600:
                return
        inflight.add(key)

    def _run() -> None:
        try:
            pdf = find_pdf(ipo_dir, company)
            text = read_pdf(pdf) if pdf else ""
            profile = extractor(company, text)
            with lock:
                cache[key] = profile
        finally:
            with lock:
                inflight.discard(key)

    threading.Thread(target=_run, daemon=True, name=f"extract-{key[:20]}").start()


def startup_pipeline(
    ipo_dir: Path,
    scrape_fn: Callable[[], int],
    trigger_fn: Callable[[str], None],
    pipeline_lock: threading.Lock,
) -> tuple[int, int]:
    if not pipeline_lock.acquire(blocking=False):
        return 0, len(list_companies(ipo_dir))

    try:
        downloaded = scrape_fn()
        companies = list_companies(ipo_dir)
        for idx, company in enumerate(companies):
            if idx > 0:
                time.sleep(3)
            trigger_fn(company)
        return downloaded, len(companies)
    finally:
        pipeline_lock.release()
