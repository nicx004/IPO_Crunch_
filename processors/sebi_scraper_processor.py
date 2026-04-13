from __future__ import annotations

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def scrape_sebi_drhps(ipo_dir: Path, cap: int = 20) -> int:
    sebi_url = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&ssid=15&smid=10"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    downloaded = 0

    try:
        response = requests.get(sebi_url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        links: list[tuple[str, str]] = []
        for anchor in soup.find_all("a", href=True):
            text = " ".join(anchor.get_text().split()).upper()
            if "DRHP" in text or "DRAFT" in text:
                company = re.sub(r"\s+", "_", text.split("-")[0].strip())
                company = re.sub(r"[^A-Za-z0-9_]", "", company) or "UNKNOWN"
                links.append((company, anchor["href"]))

        for company, href in links[:cap]:
            out_path = ipo_dir / f"{company}_DRHP.pdf"
            if out_path.exists():
                continue

            full_url = href if href.startswith("http") else "https://www.sebi.gov.in" + href
            pdf_url = full_url if full_url.lower().endswith(".pdf") else None

            if not pdf_url:
                page = requests.get(full_url, headers=headers, timeout=20)
                page_soup = BeautifulSoup(page.text, "html.parser")
                iframe = page_soup.find("iframe")
                if iframe and iframe.get("src"):
                    pdf_url = iframe["src"]
                if not pdf_url:
                    for second_anchor in page_soup.find_all("a", href=True):
                        if ".pdf" in second_anchor["href"].lower():
                            pdf_url = second_anchor["href"]
                            break

            if not pdf_url:
                continue

            if not pdf_url.startswith("http"):
                pdf_url = "https://www.sebi.gov.in" + pdf_url

            pdf_resp = requests.get(pdf_url, headers={**headers, "Referer": full_url}, timeout=30)
            content_type = (pdf_resp.headers.get("Content-Type", "") or "").lower()
            looks_like_pdf = (
                "application/pdf" in content_type
                or pdf_url.lower().endswith(".pdf")
                or pdf_resp.content[:4] == b"%PDF"
            )
            if not looks_like_pdf:
                continue

            out_path.write_bytes(pdf_resp.content)
            downloaded += 1

    except Exception:
        return downloaded

    return downloaded
