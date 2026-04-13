"""Reference processor modules split out from app.py.

These files are intentionally not imported by app.py so deployment behavior stays unchanged.
"""

from .config_processor import AppConfig, cors_origins, company_key
from .pdf_processor import read_pdf, find_pdf, list_companies
from .sebi_scraper_processor import scrape_sebi_drhps
from .llm_processor import model_candidates, is_missing_model_error, chat_with_model_fallback
from .metrics_processor import (
    format_inr,
    extract_basic_metrics_from_text,
    fill_missing_basic_metrics,
    fetch_web_snippets,
    fill_missing_from_web,
)
from .extraction_processor import EXTRACTION_PROMPT, fallback_profile, extract_profile
from .pipeline_processor import trigger_extraction, startup_pipeline
from .chat_processor import chat_with_drhp
