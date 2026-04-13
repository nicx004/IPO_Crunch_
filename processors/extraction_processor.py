from __future__ import annotations

import json
import re
import time

from openai import OpenAI

from .llm_processor import chat_with_model_fallback
from .metrics_processor import fill_missing_basic_metrics, fill_missing_from_web


EXTRACTION_PROMPT = """You are a senior sell-side IPO analyst at a top Indian investment bank.
Analyse the DRHP filing text below for {company} and return ONLY a single valid JSON object.
No markdown fences, no prose, just JSON.

Return this exact structure with IPO metrics, qualitative summary, risks, and peers.
DRHP Text:
{text}
"""


def fallback_profile(company: str) -> dict:
    return {
        "company": company,
        "sector": "Diversified",
        "headline": f"{company} - DRHP filed with SEBI.",
        "summary": "Profile is being extracted. Please check back in a moment.",
        "hero_tags": ["DRHP Filed", "SEBI", "IPO"],
        "issue_size": None,
        "price_band": None,
        "implied_value": None,
        "lot_size": None,
        "subscription_status": None,
        "grey_market": None,
        "proceeds_capex": None,
        "proceeds_debt": None,
        "proceeds_general": None,
        "proceeds_other": None,
        "thesis": "Awaiting extraction.",
        "upside": ["Extraction in progress..."],
        "watchlist": ["Extraction in progress..."],
        "briefing": "Extracting data from DRHP filing...",
        "focus_demand": "Pending analysis.",
        "focus_margins": "Pending analysis.",
        "focus_execution": "Pending analysis.",
        "signal_mood": "Mixed",
        "signal_note": "Analysis in progress.",
        "peers": ["Peer 1", "Peer 2", "Peer 3", "Peer 4"],
        "peer_ev_sales": [2.5, 3.0, 2.0, 3.5],
        "peer_pe": [25.0, 30.0, 20.0, 35.0],
        "risk_execution": 30,
        "risk_regulatory": 20,
        "risk_supply_chain": 15,
        "risk_demand": 20,
        "risk_competition": 15,
        "extracted_at": time.time(),
        "status": "pending",
    }


def extract_profile(
    client: OpenAI,
    llm_key: str,
    primary_model: str,
    fallback_models: list[str],
    company: str,
    text: str,
    enable_web_enrichment: bool = True,
) -> dict:
    if not llm_key:
        return fallback_profile(company)
    if not text.strip():
        return fallback_profile(company)

    prompt = EXTRACTION_PROMPT.format(company=company, text=text[:15000])
    try:
        raw, model_used = chat_with_model_fallback(
            client=client,
            primary_model=primary_model,
            fallback_models=fallback_models,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = re.sub(r"^```(?:json)?\\s*", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\\s*```$", "", raw).strip()
        data = json.loads(raw)
        data["company"] = company
        data["extracted_at"] = time.time()
        data["status"] = "extracted"
        data["llm_model"] = model_used
        data = fill_missing_basic_metrics(data, text)
        return fill_missing_from_web(company, data, enabled=enable_web_enrichment)
    except Exception:
        data = fill_missing_basic_metrics(fallback_profile(company), text)
        return fill_missing_from_web(company, data, enabled=enable_web_enrichment)
