from __future__ import annotations

import re

import requests
from bs4 import BeautifulSoup


def format_inr(value: str, unit: str | None = None, suffix: str = "") -> str:
    normalized_unit = (unit or "").strip().lower()
    unit_map = {
        "cr": "crore",
        "crore": "crore",
        "lakh": "lakh",
        "lakhs": "lakh",
        "million": "million",
        "billion": "billion",
    }
    display_unit = unit_map.get(normalized_unit, normalized_unit)
    base = f"INR {value}"
    if display_unit:
        base = f"{base} {display_unit}"
    if suffix:
        base = f"{base} {suffix}"
    return base


def extract_basic_metrics_from_text(text: str) -> dict[str, str]:
    compact = " ".join(text.split())
    metrics: dict[str, str] = {}

    issue_match = re.search(
        r"(?i)(?:issue size|offer size|fresh issue(?: size)?|total issue size|total offer size)"
        r".{0,80}?(?:rs\\.?|inr|₹)\\s*([0-9][0-9,]*(?:\\.[0-9]+)?)\\s*(crore|cr|lakh|lakhs|million|billion)?",
        compact,
    )
    if issue_match:
        metrics["issue_size"] = format_inr(issue_match.group(1), issue_match.group(2))

    price_band_match = re.search(
        r"(?i)(?:price band|price band of|price per equity share).{0,80}?"
        r"(?:rs\\.?|inr|₹)\\s*([0-9][0-9,]*(?:\\.[0-9]+)?)\\s*(?:-|to|–)\\s*"
        r"(?:rs\\.?|inr|₹)?\\s*([0-9][0-9,]*(?:\\.[0-9]+)?)",
        compact,
    )
    if price_band_match:
        metrics["price_band"] = f"INR {price_band_match.group(1)} - {price_band_match.group(2)} per share"

    lot_size_match = re.search(
        r"(?i)(?:lot size|minimum bid lot|market lot).{0,40}?([0-9][0-9,]{0,6})\\s*(?:equity\\s*shares|shares)",
        compact,
    )
    if lot_size_match:
        metrics["lot_size"] = f"{lot_size_match.group(1)} shares"

    implied_value_match = re.search(
        r"(?i)(?:market cap(?:italization)?|post-issue market cap|implied valuation).{0,80}?"
        r"(?:rs\\.?|inr|₹)\\s*([0-9][0-9,]*(?:\\.[0-9]+)?)\\s*(crore|cr|lakh|lakhs|million|billion)",
        compact,
    )
    if implied_value_match:
        metrics["implied_value"] = format_inr(implied_value_match.group(1), implied_value_match.group(2), "post-issue market cap")

    return metrics


def fill_missing_basic_metrics(data: dict, text: str) -> dict:
    extracted = extract_basic_metrics_from_text(text)
    missing_tokens = {
        "",
        "null",
        "none",
        "n/a",
        "awaiting data",
        "awaiting extraction.",
        "pending analysis.",
    }

    for field, value in extracted.items():
        current = data.get(field)
        current_str = str(current).strip().lower() if current is not None else ""
        if current is None or current_str in missing_tokens:
            data[field] = value

    if not data.get("subscription_status"):
        data["subscription_status"] = "Not yet open"

    return data


def fetch_web_snippets(company: str, enabled: bool = True) -> str:
    if not enabled:
        return ""

    query = f"{company} IPO grey market premium GMP subscription status price band lot size"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    try:
        resp = requests.get("https://duckduckgo.com/html/", params={"q": query}, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        snippets = [
            " ".join(sn.get_text(" ").split())
            for sn in soup.select(".result__snippet")[:8]
            if sn.get_text(strip=True)
        ]
        return "\n".join(snippets)
    except Exception:
        return ""


def fill_missing_from_web(company: str, data: dict, enabled: bool = True) -> dict:
    missing_fields = [
        field for field in ("issue_size", "price_band", "lot_size", "implied_value", "grey_market")
        if not data.get(field)
    ]
    if not missing_fields and data.get("subscription_status") not in (None, "", "Not yet open"):
        return data

    snippets = fetch_web_snippets(company, enabled=enabled)
    if not snippets:
        return data

    web_metrics = extract_basic_metrics_from_text(snippets)
    for field, value in web_metrics.items():
        if not data.get(field) and value:
            data[field] = value

    if not data.get("grey_market"):
        gmp_match = re.search(
            r"(?i)(?:gmp|grey\\s*market\\s*premium).{0,30}?(?:rs\\.?|inr|₹)?\\s*([+-]?[0-9][0-9,]*(?:\\.[0-9]+)?)",
            snippets,
        )
        if gmp_match:
            data["grey_market"] = f"INR {gmp_match.group(1)} premium"

    sub_text = data.get("subscription_status")
    if not sub_text or sub_text == "Not yet open":
        sub_match = re.search(
            r"(?i)([0-9]+(?:\\.[0-9]+)?)\\s*x\\s*(?:subscribed|subscription)",
            snippets,
        )
        if not sub_match:
            sub_match = re.search(
                r"(?i)(?:subscribed|subscription).{0,20}?([0-9]+(?:\\.[0-9]+)?)\\s*x",
                snippets,
            )
        if sub_match:
            data["subscription_status"] = f"{sub_match.group(1)}x subscribed"

    return data
