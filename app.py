"""
IPO Crunch — Real-time DRHP Analyser
FastAPI backend + vanilla JS frontend. Zero Gradio. Deploys on Render reliably.

Architecture:
  - FastAPI serves HTML SPA + REST API
  - SEBI scraper downloads latest DRHPs on startup and daily
  - PDF text is read directly (no vectorstore needed)
  - LLM (OpenRouter/OpenAI) extracts structured data in one call
  - Results cached in memory, refreshed when new PDFs arrive
"""

from __future__ import annotations
from contextlib import asynccontextmanager
import json, os, re, threading, time
from pathlib import Path
from typing import Optional
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from openai import OpenAI
from fastapi import FastAPI, BackgroundTasks, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
IPO_DIR = Path(os.getenv("IPO_DATA_DIR", "./ipo_data"))
IPO_DIR.mkdir(exist_ok=True)

LLM_KEY    = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
LLM_MODEL  = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free")
LLM_BASE   = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
PORT       = int(os.getenv("PORT", 7860))
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN", "").strip()

llm = OpenAI(api_key=LLM_KEY or "sk-no-key", base_url=LLM_BASE, timeout=90)

# ══════════════════════════════════════════════════════════════════════════════
# In-memory cache
# ══════════════════════════════════════════════════════════════════════════════
CACHE: dict[str, dict] = {}          # company_key → profile dict
INFLIGHT: set[str] = set()           # company keys currently being extracted
LOCK = threading.Lock()
STARTUP_DONE = False
PIPELINE_LOCK = threading.Lock()
LAST_STARTUP_STATUS = "not_started"


def _cors_origins() -> list[str]:
  raw = os.getenv("CORS_ORIGINS", "").strip()
  if raw:
    return [o.strip() for o in raw.split(",") if o.strip()]

  render_external = os.getenv("RENDER_EXTERNAL_URL", "").strip()
  if render_external:
    if render_external.startswith("http://") or render_external.startswith("https://"):
      return [render_external]
    return [f"https://{render_external}"]

  # Local/dev fallback.
  return ["*"]


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


# ══════════════════════════════════════════════════════════════════════════════
# PDF utilities
# ══════════════════════════════════════════════════════════════════════════════
def read_pdf(path: Path, max_chars: int = 18000) -> str:
    """Read PDF pages directly — no vectorstore needed."""
    try:
        reader = PdfReader(str(path))
        parts = []
        # Priority: first 12 pages (cover, summary, objects of issue, price band)
        # then pages 12-40 (risk factors, financials, business overview)
        priority = list(reader.pages[:12]) + list(reader.pages[12:40])
        for page in priority:
            t = (page.extract_text() or "").strip()
            if t:
                parts.append(t)
        text = "\n\n".join(parts)
        return text[:max_chars]
    except Exception as e:
        print(f"PDF read error {path}: {e}")
        return ""


def find_pdf(company: str) -> Optional[Path]:
    target = _key(company)
    best: Optional[tuple[int, Path]] = None
    for p in IPO_DIR.glob("*.pdf"):
        stem = re.sub(r"_DRHP$", "", p.stem, flags=re.IGNORECASE).replace("_", " ")
        cand = _key(stem)
        if cand == target:
            return p
        # overlap scoring
        t_tok = set(target.split())
        c_tok = set(cand.split())
        overlap = len(t_tok & c_tok)
        if overlap >= max(2, min(len(t_tok), 3)):
            score = overlap * 100 - abs(len(c_tok) - len(t_tok))
            if best is None or score > best[0]:
                best = (score, p)
    return best[1] if best else None


def list_companies() -> list[str]:
    companies = []
    for p in sorted(IPO_DIR.glob("*.pdf")):
        stem = re.sub(r"_DRHP$", "", p.stem, flags=re.IGNORECASE)
        name = stem.replace("_", " ").strip().title()
        if name:
            companies.append(name)
    return sorted(set(companies))


# ══════════════════════════════════════════════════════════════════════════════
# SEBI Scraper
# ══════════════════════════════════════════════════════════════════════════════
def scrape_sebi_drhps() -> int:
    """Download latest DRHPs from SEBI. Returns count of new files."""
    SEBI_URL = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&ssid=15&smid=10"
    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    downloaded = 0

    try:
        print("Fetching SEBI DRHP listings...")
        resp = requests.get(SEBI_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            text = " ".join(a.get_text().split()).upper()
            if "DRHP" in text or "DRAFT" in text:
                company = re.sub(r"\s+", "_", text.split("-")[0].strip())
                company = re.sub(r"[^A-Za-z0-9_]", "", company) or "UNKNOWN"
                links.append((company, a["href"]))

        print(f"Found {len(links)} DRHP listings")

        for company, href in links[:20]:  # cap at 20 to avoid timeout
            out_path = IPO_DIR / f"{company}_DRHP.pdf"
            if out_path.exists():
                continue
            try:
                full_url = href if href.startswith("http") else "https://www.sebi.gov.in" + href
                pdf_url = full_url if full_url.lower().endswith(".pdf") else None
                if not pdf_url:
                  page = requests.get(full_url, headers=HEADERS, timeout=20)
                  page_soup = BeautifulSoup(page.text, "html.parser")
                  iframe = page_soup.find("iframe")
                  if iframe and iframe.get("src"):
                    pdf_url = iframe["src"]
                  if not pdf_url:
                    for a2 in page_soup.find_all("a", href=True):
                      if ".pdf" in a2["href"].lower():
                        pdf_url = a2["href"]
                        break
                if not pdf_url:
                    continue
                if not pdf_url.startswith("http"):
                    pdf_url = "https://www.sebi.gov.in" + pdf_url
                pdf_resp = requests.get(pdf_url, headers={**HEADERS, "Referer": full_url}, timeout=30)
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
                print(f"Downloaded: {out_path.name}")
            except Exception as e:
                print(f"Failed {company}: {e}")

        print(f"Scraping done. Downloaded {downloaded} new DRHPs.")
    except Exception as e:
        print(f"SEBI scrape failed: {e}")

    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
# LLM Extraction
# ══════════════════════════════════════════════════════════════════════════════
EXTRACTION_PROMPT = """You are a senior sell-side IPO analyst at a top Indian investment bank.
Analyse the DRHP filing text below for {company} and return ONLY a single valid JSON object.
No markdown fences, no prose, just JSON.

Return this exact structure:
{{
  "sector": "one of: Financial Services | Consumer Internet | Healthcare | Technology | Manufacturing | Consumer/Retail | Auto/Mobility | Energy | Industrial/Infra | Diversified",
  "headline": "one sharp analyst sentence summarising this IPO opportunity",
  "summary": "2-3 sentences: what the company does, its market position, and why this IPO matters",
  "hero_tags": ["tag1", "tag2", "tag3"],
  "issue_size": "INR X crore (null if not found)",
  "price_band": "INR X - Y per share (null if not found)",
  "implied_value": "INR X crore post-issue market cap (null if not found)",
  "lot_size": "X shares (null if not found)",
  "subscription_status": "Not yet open / X.Xx subscribed (null if not found)",
  "grey_market": "INR X premium (null if not found)",
  "proceeds_capex": "INR X crore for capital expenditure (null if not found)",
  "proceeds_debt": "INR X crore for debt repayment (null if not found)",
  "proceeds_general": "INR X crore for general corporate purposes (null if not found)",
  "proceeds_other": "INR X crore for other stated use (null if not found)",
  "thesis": "one sentence: the core investment thesis for a fund manager",
  "upside": ["bullet 1 — key upside driver", "bullet 2", "bullet 3"],
  "watchlist": ["bullet 1 — key risk", "bullet 2", "bullet 3"],
  "briefing": "3-4 sentence analyst briefing note covering business model, financials, and IPO rationale",
  "focus_demand": "assessment of demand visibility and market size",
  "focus_margins": "assessment of margin profile and profitability trajectory",
  "focus_execution": "key execution risks and dependencies",
  "signal_mood": "Constructive | Cautious | Mixed | Positive | Negative",
  "signal_note": "one sentence on current market sentiment for this IPO",
  "peers": ["Peer Company 1", "Peer Company 2", "Peer Company 3", "Peer Company 4"],
  "peer_ev_sales": [2.5, 3.1, 1.8, 4.2],
  "peer_pe": [25.0, 30.5, 18.0, 42.0],
  "risk_execution": 30,
  "risk_regulatory": 20,
  "risk_supply_chain": 15,
  "risk_demand": 20,
  "risk_competition": 15
}}

Rules:
- Financial numbers: extract ONLY what is explicitly in the text. Use null if not found.
- Qualitative fields (sector, thesis, upside, etc.): use your analyst expertise if text is sparse.
- peers: use real NSE/BSE listed companies in the same sector.
- peer_ev_sales: realistic EV/Sales multiples (0.5-10x range).
- peer_pe: realistic P/E ratios (10-80x range).
- risk_* fields: integers that sum to exactly 100. Weight by how much the filing discusses each.
- hero_tags: 3 concise labels (e.g. "EV Charging", "High Growth", "Pre-Profit").

DRHP Text:
{text}
"""


def extract_profile(company: str, text: str) -> dict:
    """Single LLM call — extracts all fields at once."""
    if not LLM_KEY:
        return _fallback_profile(company)
    if not text.strip():
        return _fallback_profile(company)

    prompt = EXTRACTION_PROMPT.format(company=company, text=text[:15000])
    try:
        resp = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content or ""
        # Strip markdown fences if model adds them
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw).strip()
        data = json.loads(raw)
        data["company"] = company
        data["extracted_at"] = time.time()
        data["status"] = "extracted"
        return data
    except json.JSONDecodeError as e:
        print(f"JSON parse failed for {company}: {e}")
        return _fallback_profile(company)
    except Exception as e:
        print(f"LLM extraction failed for {company}: {e}")
        return _fallback_profile(company)


def _fallback_profile(company: str) -> dict:
    return {
        "company": company,
        "sector": "Diversified",
        "headline": f"{company} — DRHP filed with SEBI.",
        "summary": "Profile is being extracted. Please check back in a moment.",
        "hero_tags": ["DRHP Filed", "SEBI", "IPO"],
        "issue_size": None, "price_band": None, "implied_value": None,
        "lot_size": None, "subscription_status": None, "grey_market": None,
        "proceeds_capex": None, "proceeds_debt": None,
        "proceeds_general": None, "proceeds_other": None,
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
        "risk_execution": 30, "risk_regulatory": 20,
        "risk_supply_chain": 15, "risk_demand": 20, "risk_competition": 15,
        "extracted_at": time.time(), "status": "pending",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Background extraction
# ══════════════════════════════════════════════════════════════════════════════
def trigger_extraction(company: str, force: bool = False) -> None:
    """Start async extraction for a company if not already cached/inflight."""
    k = _key(company)
    with LOCK:
        if k in INFLIGHT:
            return
        if not force and k in CACHE and CACHE[k].get("status") == "extracted":
            age = time.time() - CACHE[k].get("extracted_at", 0)
            if age < 3600:  # 1 hour TTL
                return
        INFLIGHT.add(k)

    def _run():
        try:
            pdf = find_pdf(company)
            text = read_pdf(pdf) if pdf else ""
            print(f"Extracting {company} ({len(text)} chars)...")
            profile = extract_profile(company, text)
            with LOCK:
                CACHE[k] = profile
            print(f"✅ {company} extracted")
        except Exception as e:
            print(f"Extraction error {company}: {e}")
        finally:
            with LOCK:
                INFLIGHT.discard(k)

    threading.Thread(target=_run, daemon=True, name=f"extract-{k[:20]}").start()


def startup_pipeline() -> None:
  """Full startup: scrape → extract all companies."""
  global STARTUP_DONE, LAST_STARTUP_STATUS

  if not PIPELINE_LOCK.acquire(blocking=False):
    print("Startup pipeline already running; skipping duplicate trigger")
    return

  print("=== Startup pipeline starting ===")
  LAST_STARTUP_STATUS = "running"
  try:
    # 1. Download fresh DRHPs from SEBI
    downloaded = scrape_sebi_drhps()
    # 2. Extract all companies found (staggered to avoid API rate limits)
    companies = list_companies()
    print(f"Starting extraction for {len(companies)} companies...")
    for i, company in enumerate(companies):
      if i > 0:
        time.sleep(3)
      trigger_extraction(company)
    STARTUP_DONE = True
    LAST_STARTUP_STATUS = f"complete: downloaded={downloaded}, companies={len(companies)}"
    print("=== Startup pipeline complete ===")
  except Exception as e:
    LAST_STARTUP_STATUS = f"error: {e}"
    raise
  finally:
    PIPELINE_LOCK.release()


# ══════════════════════════════════════════════════════════════════════════════
# Chat / RAG
# ══════════════════════════════════════════════════════════════════════════════
def chat_with_drhp(company: str, message: str, history: list[dict]) -> str:
    """Answer questions about a company's DRHP."""
    if not LLM_KEY:
        return "⚠️ No API key configured. Set OPENROUTER_API_KEY in Render environment variables."

    pdf = find_pdf(company)
    if not pdf:
        return f"No DRHP file found for {company}. The scraper may still be running — please try again in a minute."

    context = read_pdf(pdf, max_chars=10000)

    system = f"""You are an expert IPO analyst for {company}.
You have access to the company's DRHP (Draft Red Herring Prospectus) filing.

Answer questions directly and specifically using the DRHP context provided.
- Lead with a direct answer
- Add "From the DRHP:" bullets when quoting filing evidence
- Add "Analyst view:" for reasoned inferences beyond the text
- Be concise and professional
- If something is truly not in the filing, say so clearly

DRHP Context (first {len(context)} characters of filing):
{context}"""

    messages = [{"role": "system", "content": system}]
    for h in history[-6:]:  # last 3 turns
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    try:
        resp = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
        )
        return resp.choices[0].message.content or "No response generated."
    except Exception as e:
        return f"Error: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(_: FastAPI):
    if not scheduler.running:
        scheduler.start()
    threading.Thread(target=startup_pipeline, daemon=True, name="startup").start()
    try:
        yield
    finally:
        if scheduler.running:
            scheduler.shutdown(wait=False)


app = FastAPI(title="IPO Crunch API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins(),
                   allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
  company: str
  message: str
  history: list[dict] = Field(default_factory=list)


@app.get("/api/health")
def health():
    return {"status": "ok", "startup_done": STARTUP_DONE,
      "companies": len(list_companies()), "cached": len(CACHE),
      "refresh_protected": bool(REFRESH_TOKEN),
      "startup_status": LAST_STARTUP_STATUS}


@app.get("/api/companies")
def companies():
    return {"companies": list_companies()}


@app.get("/api/profile/{company}")
def profile(company: str, background_tasks: BackgroundTasks):
    k = _key(company)
    with LOCK:
        cached = CACHE.get(k)

    if cached and cached.get("status") == "extracted":
        return cached

    # Trigger extraction if not already running
    background_tasks.add_task(trigger_extraction, company)

    if cached:
        return cached  # return pending profile while extraction runs

    # Return fallback immediately
    fp = _fallback_profile(company)
    with LOCK:
        CACHE[k] = fp
    return fp


@app.post("/api/chat")
def chat(req: ChatRequest):
    answer = chat_with_drhp(req.company, req.message, req.history)
    return {"answer": answer}


@app.post("/api/refresh")
def refresh(
  background_tasks: BackgroundTasks,
  x_refresh_token: str | None = Header(default=None, alias="X-Refresh-Token"),
):
  incoming = (x_refresh_token or "").strip()
  if REFRESH_TOKEN and incoming != REFRESH_TOKEN:
    raise HTTPException(status_code=401, detail="Unauthorized")
  background_tasks.add_task(startup_pipeline)
  return {"status": "refresh triggered"}


# ══════════════════════════════════════════════════════════════════════════════
# Frontend HTML — single-page app
# ══════════════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IPO Crunch — Real-time DRHP Analyser</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #05070a;
  --surface: rgba(10,13,18,0.97);
  --border: rgba(91,108,128,0.4);
  --text: #f2f4f8;
  --muted: #9aa7b7;
  --accent: #ffb347;
  --accent2: #ffe082;
  --accent3: #78b7ff;
  --accent4: #5ad08c;
  --red: #ff6b6b;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  background: linear-gradient(rgba(255,179,71,.025) 1px,transparent 1px),
              linear-gradient(90deg,rgba(255,179,71,.025) 1px,transparent 1px),
              linear-gradient(180deg,#05070a 0%,#080b10 100%);
  background-size: 48px 48px, 48px 48px, 100% 100%;
  color: var(--text);
  font-family: 'IBM Plex Sans', sans-serif;
  min-height: 100vh;
  line-height: 1.6;
}
.container { max-width: 1440px; margin: 0 auto; padding: 20px; }

/* ── Top bar ── */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 20px;
  background: rgba(5,7,10,.95);
  border-bottom: 1px solid var(--border);
  position: sticky; top: 0; z-index: 100;
}
.topbar-logo { color: var(--accent); font-family: 'IBM Plex Mono', monospace; font-size: 13px; letter-spacing: .12em; }
.topbar-right { display: flex; align-items: center; gap: 12px; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #555; }
.status-dot.live { background: var(--accent4); box-shadow: 0 0 6px var(--accent4); animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
select#company-select {
  background: rgba(0,0,0,.5); border: 1px solid var(--border);
  color: var(--text); padding: 8px 14px; border-radius: 4px;
  font-family: 'IBM Plex Mono', monospace; font-size: 12px;
  cursor: pointer; min-width: 220px;
}
.btn {
  background: linear-gradient(180deg,#ffbf5e,#ff9f1a);
  border: none; border-radius: 4px; color: #05070a;
  font-family: 'IBM Plex Mono', monospace; font-size: 11px;
  font-weight: 700; padding: 8px 16px; cursor: pointer; letter-spacing: .08em;
  text-transform: uppercase;
}
.btn:hover { opacity: .9; }
.btn-ghost {
  background: transparent; border: 1px solid var(--border);
  color: var(--muted);
}

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
  position: relative; overflow: hidden;
}
.card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--accent), transparent);
}
.kicker {
  color: var(--accent); font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; letter-spacing: .16em; text-transform: uppercase; margin-bottom: 6px;
}
.card-title { font-family: 'IBM Plex Mono', monospace; font-size: 18px; letter-spacing: .04em; text-transform: uppercase; margin-bottom: 8px; }
.card-copy { color: var(--muted); font-size: 13px; line-height: 1.6; }

/* ── Hero ── */
.hero {
  display: grid; grid-template-columns: 1fr 380px; gap: 20px;
  margin-bottom: 20px;
}
.hero-main {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 24px; position: relative; overflow: hidden;
}
.hero-main::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background: linear-gradient(90deg,var(--accent),var(--accent3)); }
.hero-terminal {
  display: flex; justify-content: space-between; align-items: center;
  background: rgba(0,0,0,.4); border: 1px solid var(--border);
  padding: 8px 12px; border-radius: 4px; margin-bottom: 16px;
  font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--accent);
  letter-spacing: .12em;
}
.hero-sector { color: var(--accent2); font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: .14em; text-transform: uppercase; margin-bottom: 6px; }
.hero-name { font-family: 'IBM Plex Mono', monospace; font-size: clamp(28px,4vw,44px); font-weight: 600; text-transform: uppercase; line-height: 1; margin-bottom: 10px; letter-spacing: .03em; }
.hero-headline { font-size: 15px; color: var(--text); margin-bottom: 12px; line-height: 1.5; }
.hero-summary { font-size: 13px; color: var(--muted); line-height: 1.7; max-width: 600px; margin-bottom: 16px; }
.tags { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.tag {
  background: rgba(0,0,0,.3); border: 1px solid var(--border);
  color: var(--accent); font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; letter-spacing: .1em; padding: 5px 10px; border-radius: 3px; text-transform: uppercase;
}
.focus-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-top: 16px; }
.focus-card {
  background: rgba(0,0,0,.25); border: 1px solid var(--border);
  border-radius: 6px; padding: 12px;
}
.focus-label { color: var(--accent); font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: .12em; text-transform: uppercase; display: block; margin-bottom: 6px; }
.focus-value { font-size: 13px; color: var(--text); line-height: 1.4; }

/* ── Signal panel ── */
.signal-panel { display: flex; flex-direction: column; gap: 12px; }
.signal-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 18px; }
.signal-mood {
  font-family: 'IBM Plex Mono', monospace; font-size: 28px;
  font-weight: 600; text-transform: uppercase; color: var(--accent); letter-spacing: .04em;
}
.signal-note { color: var(--muted); font-size: 13px; margin-top: 6px; line-height: 1.5; }
.signal-metrics { display: grid; gap: 8px; margin-top: 12px; }
.signal-metric {
  display: flex; justify-content: space-between; align-items: center;
  background: rgba(0,0,0,.25); border: 1px solid var(--border);
  padding: 8px 12px; border-radius: 4px;
}
.signal-metric-label { color: var(--muted); font-size: 13px; }
.signal-metric-val { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--text); }
.thesis-box {
  background: rgba(10,18,28,.9); border: 1px solid rgba(120,183,255,.25);
  border-left: 3px solid var(--accent3); border-radius: 4px; padding: 14px;
}
.thesis-label { color: var(--accent2); font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: .14em; text-transform: uppercase; margin-bottom: 6px; }
.thesis-text { color: var(--text); font-size: 13px; line-height: 1.7; }

/* ── Metrics grid ── */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 10px; margin-bottom: 20px;
}
.metric-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
  padding: 14px; position: relative; overflow: hidden; min-height: 110px;
}
.metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
.metric-card.amber::before { background: var(--accent); }
.metric-card.yellow::before { background: var(--accent2); }
.metric-card.blue::before { background: var(--accent3); }
.metric-card.green::before { background: var(--accent4); }
.metric-label { color: var(--muted); font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: .12em; text-transform: uppercase; margin-bottom: 14px; display: block; }
.metric-value { color: var(--accent); font-family: 'IBM Plex Mono', monospace; font-size: 18px; font-weight: 600; line-height: 1.1; margin-bottom: 8px; text-transform: uppercase; }
.metric-note { color: var(--muted); font-size: 11px; line-height: 1.4; }

/* ── Charts ── */
.charts-row { display: grid; grid-template-columns: 3fr 2fr; gap: 20px; margin-bottom: 20px; }
.chart-container { position: relative; height: 280px; }

/* ── Briefing ── */
.briefing-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
.briefing-section h4 { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }
.briefing-list { list-style: none; }
.briefing-list li { color: var(--text); font-size: 13px; line-height: 1.6; padding: 8px 0; border-bottom: 1px solid rgba(91,108,128,.15); padding-left: 14px; position: relative; }
.briefing-list li::before { content: '›'; position: absolute; left: 0; color: var(--accent); }
.briefing-list li.risk::before { color: var(--red); }
.proceeds-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
.proceeds-table th { color: var(--muted); font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: .1em; text-transform: uppercase; padding: 8px 0; border-bottom: 1px solid var(--border); text-align: left; }
.proceeds-table td { color: var(--text); font-size: 13px; padding: 10px 0; border-bottom: 1px solid rgba(91,108,128,.12); }
.briefing-text { color: var(--muted); font-size: 13px; line-height: 1.75; }

/* ── Chat ── */
.chat-section { margin-bottom: 30px; }
.chat-messages {
  height: 360px; overflow-y: auto;
  background: rgba(5,7,10,.85); border: 1px solid var(--border);
  border-radius: 6px; padding: 16px; margin-bottom: 12px;
  display: flex; flex-direction: column; gap: 12px;
}
.msg { max-width: 85%; }
.msg.user { align-self: flex-end; }
.msg.assistant { align-self: flex-start; }
.msg-bubble {
  padding: 12px 16px; border-radius: 14px; font-size: 14px; line-height: 1.6;
}
.msg.user .msg-bubble { background: rgba(16,33,52,.9); border: 1px solid rgba(120,183,255,.2); }
.msg.assistant .msg-bubble { background: rgba(12,12,12,.95); border: 1px solid rgba(255,179,71,.18); color: var(--text); }
.chat-input-row { display: flex; gap: 10px; }
.chat-input {
  flex: 1; background: rgba(5,7,10,.9); border: 1px solid var(--border);
  color: var(--text); padding: 12px 16px; border-radius: 4px;
  font-family: 'IBM Plex Mono', monospace; font-size: 13px;
  outline: none;
}
.chat-input:focus { border-color: rgba(255,179,71,.5); }
.quick-prompts { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
.quick-btn {
  background: rgba(5,7,10,.9); border: 1px solid var(--border);
  color: var(--accent); font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; letter-spacing: .08em; padding: 7px 12px;
  border-radius: 3px; cursor: pointer; text-transform: uppercase;
}
.quick-btn:hover { border-color: rgba(255,179,71,.5); background: rgba(32,24,6,.9); }

/* ── Loader ── */
.skeleton { background: linear-gradient(90deg,rgba(255,255,255,.04) 25%,rgba(255,255,255,.08) 50%,rgba(255,255,255,.04) 75%); background-size: 200% 100%; animation: shimmer 1.5s infinite; border-radius: 3px; }
@keyframes shimmer { from{background-position:200% 0} to{background-position:-200% 0} }
.loading-val { display: inline-block; width: 80px; height: 14px; }

/* ── Responsive ── */
@media(max-width:1100px) { .metrics-grid{grid-template-columns:repeat(3,1fr)} }
@media(max-width:900px) {
  .hero{grid-template-columns:1fr}
  .charts-row{grid-template-columns:1fr}
  .briefing-row{grid-template-columns:1fr}
  .focus-grid{grid-template-columns:1fr}
}
@media(max-width:600px) { .metrics-grid{grid-template-columns:repeat(2,1fr)} }

/* ── Footer ── */
footer { text-align: center; color: #3a4a5a; font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: .1em; padding: 20px; text-transform: uppercase; }
</style>
</head>
<body>

<nav class="topbar">
  <div class="topbar-logo">⬡ IPO CRUNCH :: DRHP INTELLIGENCE</div>
  <div class="topbar-right">
    <div class="status-dot" id="status-dot"></div>
    <span id="status-text" style="color:var(--muted);font-family:'IBM Plex Mono',monospace;font-size:11px">Loading...</span>
    <select id="company-select"><option>Loading companies...</option></select>
    <button class="btn" onclick="triggerRefresh()">↻ Refresh Data</button>
  </div>
</nav>

<div class="container">

  <!-- Hero -->
  <div class="hero" style="margin-top:20px">
    <div class="hero-main">
      <div class="hero-terminal">
        <span>IPO CRUNCH :: ACTIVE FILING</span>
        <span id="hero-meta">—</span>
      </div>
      <div class="hero-sector kicker" id="hero-sector">—</div>
      <div class="hero-name" id="hero-name">Select a Company</div>
      <div class="hero-headline" id="hero-headline">Choose a company from the dropdown above to load its DRHP analysis.</div>
      <div class="hero-summary" id="hero-summary"></div>
      <div class="tags" id="hero-tags"></div>
      <div class="focus-grid" id="focus-grid"></div>
    </div>
    <div class="signal-panel">
      <div class="signal-card">
        <div class="kicker">[Market signal]</div>
        <div class="signal-mood" id="signal-mood">—</div>
        <div class="signal-note" id="signal-note"></div>
        <div class="signal-metrics" id="signal-metrics"></div>
      </div>
      <div class="card">
        <div class="thesis-label kicker">Deal thesis</div>
        <div class="thesis-text" id="thesis-text">—</div>
      </div>
    </div>
  </div>

  <!-- Metrics -->
  <div class="card" style="margin-bottom:20px">
    <div class="kicker" style="margin-bottom:14px">[Live deal snapshot] Primary Market Pulse</div>
    <div class="metrics-grid" id="metrics-grid">
      <div class="metric-card amber"><span class="metric-label">Issue Size</span><div class="metric-value skeleton loading-val"></div></div>
      <div class="metric-card yellow"><span class="metric-label">Price Band</span><div class="metric-value skeleton loading-val"></div></div>
      <div class="metric-card blue"><span class="metric-label">Implied Value</span><div class="metric-value skeleton loading-val"></div></div>
      <div class="metric-card green"><span class="metric-label">Lot Size</span><div class="metric-value skeleton loading-val"></div></div>
      <div class="metric-card amber"><span class="metric-label">Subscription</span><div class="metric-value skeleton loading-val"></div></div>
      <div class="metric-card yellow"><span class="metric-label">Grey Market</span><div class="metric-value skeleton loading-val"></div></div>
    </div>
  </div>

  <!-- Charts -->
  <div class="charts-row">
    <div class="card">
      <div class="kicker" style="margin-bottom:12px">[Market map] Relative Multiple View</div>
      <div class="chart-container"><canvas id="peer-chart"></canvas></div>
    </div>
    <div class="card">
      <div class="kicker" style="margin-bottom:12px">[Risk stack] Filing Risk Emphasis</div>
      <div class="chart-container"><canvas id="risk-chart"></canvas></div>
    </div>
  </div>

  <!-- Briefing -->
  <div class="card" style="margin-bottom:20px">
    <div class="kicker" style="margin-bottom:16px">[Deal briefing] Analyst Summary</div>
    <p class="briefing-text" id="briefing-text" style="margin-bottom:20px">—</p>
    <div class="briefing-row">
      <div>
        <div class="briefing-section">
          <h4>Upside Drivers</h4>
          <ul class="briefing-list" id="upside-list"></ul>
        </div>
        <div class="briefing-section" style="margin-top:16px">
          <h4>Watch List</h4>
          <ul class="briefing-list" id="watchlist-list"></ul>
        </div>
      </div>
      <div>
        <div class="kicker" style="margin-bottom:10px">Use of Proceeds</div>
        <table class="proceeds-table">
          <thead><tr><th>Bucket</th><th>Allocation</th></tr></thead>
          <tbody id="proceeds-body"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Chat -->
  <div class="card chat-section">
    <div class="kicker" style="margin-bottom:12px">[AI Workspace] Ask the DRHP</div>
    <div class="quick-prompts">
      <button class="quick-btn" onclick="quickPrompt('What is the use of proceeds for this IPO?')">Use of Proceeds</button>
      <button class="quick-btn" onclick="quickPrompt('Summarize the top risk factors for this IPO.')">Risk Factors</button>
      <button class="quick-btn" onclick="quickPrompt('What are the key financial metrics and profitability?')">Financial Quality</button>
      <button class="quick-btn" onclick="quickPrompt('Tell me about the management team and promoter background.')">Management</button>
      <button class="quick-btn" onclick="quickPrompt('Who are the main competitors and what is the competitive landscape?')">Competition</button>
      <button class="quick-btn" onclick="quickPrompt('What is the company business model and revenue streams?')">Business Model</button>
    </div>
    <div class="chat-messages" id="chat-messages">
      <div class="msg assistant">
        <div class="msg-bubble">Select a company and ask me anything about its DRHP filing — financial metrics, risk factors, use of proceeds, management background, competitive landscape, and more.</div>
      </div>
    </div>
    <div class="chat-input-row">
      <input class="chat-input" id="chat-input" type="text"
             placeholder="Ask anything about this DRHP..."
             onkeydown="if(event.key==='Enter')sendChat()">
      <button class="btn" onclick="sendChat()">Run Analysis</button>
      <button class="btn btn-ghost" onclick="clearChat()">Reset</button>
    </div>
  </div>

</div>
<footer>IPO Crunch · Real-time DRHP Intelligence · Not investment advice · Data sourced from SEBI public filings</footer>

<script>
const API = '';
let currentCompany = '';
let chatHistory = [];
let peerChart = null;
let riskChart = null;
let pollInterval = null;

// ── Init ──────────────────────────────────────────────────────────────────
async function init() {
  await loadCompanies();
  checkStatus();
  setInterval(checkStatus, 15000);
}

async function checkStatus() {
  try {
    const r = await fetch(`${API}/api/health`);
    const d = await r.json();
    const dot = document.getElementById('status-dot');
    const txt = document.getElementById('status-text');
    dot.className = 'status-dot' + (d.companies > 0 ? ' live' : '');
    txt.textContent = d.companies > 0
      ? `${d.companies} companies · ${d.cached} cached`
      : (d.startup_done ? 'Ready' : 'Loading...');
  } catch(e) {}
}

async function loadCompanies() {
  try {
    const r = await fetch(`${API}/api/companies`);
    const d = await r.json();
    const sel = document.getElementById('company-select');
    sel.innerHTML = '';
    if (!d.companies || d.companies.length === 0) {
      sel.innerHTML = '<option value="">No companies yet — scraping SEBI...</option>';
      setTimeout(loadCompanies, 10000);
      return;
    }
    sel.innerHTML = '<option value="">— Select IPO —</option>' +
      d.companies.map(c => `<option value="${c}">${c}</option>`).join('');
    sel.onchange = () => { if(sel.value) loadCompany(sel.value); };
    // Auto-load first
    if (d.companies.length > 0) loadCompany(d.companies[0]);
  } catch(e) {
    setTimeout(loadCompanies, 5000);
  }
}

// ── Load company ──────────────────────────────────────────────────────────
async function loadCompany(name) {
  currentCompany = name;
  chatHistory = [];
  clearChat();
  setLoadingState(name);
  if (pollInterval) clearInterval(pollInterval);

  const profile = await fetchProfile(name);
  renderProfile(profile);

  // Poll if still pending
  if (profile.status === 'pending') {
    pollInterval = setInterval(async () => {
      const p = await fetchProfile(name);
      if (p.status === 'extracted') {
        clearInterval(pollInterval);
        renderProfile(p);
      }
    }, 5000);
  }
}

async function fetchProfile(name) {
  try {
    const r = await fetch(`${API}/api/profile/${encodeURIComponent(name)}`);
    return await r.json();
  } catch(e) {
    return { status: 'error', company: name };
  }
}

function setLoadingState(name) {
  document.getElementById('hero-name').textContent = name;
  document.getElementById('hero-sector').textContent = 'Extracting from DRHP...';
  document.getElementById('hero-headline').textContent = 'Analysing filing — please wait...';
  document.getElementById('hero-summary').textContent = '';
  document.getElementById('hero-tags').innerHTML = '';
  document.getElementById('focus-grid').innerHTML = '';
  document.getElementById('signal-mood').textContent = '—';
  document.getElementById('signal-note').textContent = '';
  document.getElementById('thesis-text').textContent = '—';
  document.getElementById('briefing-text').textContent = 'Extracting...';
  document.getElementById('upside-list').innerHTML = '';
  document.getElementById('watchlist-list').innerHTML = '';
  document.getElementById('proceeds-body').innerHTML = '';
  setMetricLoading();
}

function setMetricLoading() {
  const metrics = [
    {label:'Issue Size',color:'amber'},{label:'Price Band',color:'yellow'},
    {label:'Implied Value',color:'blue'},{label:'Lot Size',color:'green'},
    {label:'Subscription',color:'amber'},{label:'Grey Market',color:'yellow'},
  ];
  document.getElementById('metrics-grid').innerHTML = metrics.map(m => `
    <div class="metric-card ${m.color}">
      <span class="metric-label">${m.label}</span>
      <div class="metric-value skeleton loading-val" style="height:22px;width:90px"></div>
      <div class="metric-note skeleton" style="height:11px;width:60px;margin-top:8px"></div>
    </div>`).join('');
}

// ── Render ────────────────────────────────────────────────────────────────
function renderProfile(p) {
  if (!p || !p.company) return;

  // Hero
  document.getElementById('hero-name').textContent = p.company.toUpperCase();
  document.getElementById('hero-sector').textContent = `[${p.sector || '—'}]`;
  document.getElementById('hero-headline').textContent = p.headline || '—';
  document.getElementById('hero-summary').textContent = p.summary || '';
  document.getElementById('hero-meta').textContent = `${p.sector || '—'} / NSE / BSE / DRHP Filing`;

  // Tags
  const tags = p.hero_tags || [];
  document.getElementById('hero-tags').innerHTML = tags.map(t =>
    `<span class="tag">${t}</span>`).join('');

  // Focus cards
  const focuses = [
    {label:'Demand Signal', value: p.focus_demand},
    {label:'Margin Lens',   value: p.focus_margins},
    {label:'Execution',     value: p.focus_execution},
  ];
  document.getElementById('focus-grid').innerHTML = focuses.map(f => `
    <div class="focus-card">
      <span class="focus-label">${f.label}</span>
      <div class="focus-value">${f.value || 'Pending analysis...'}</div>
    </div>`).join('');

  // Signal
  document.getElementById('signal-mood').textContent = p.signal_mood || '—';
  document.getElementById('signal-note').textContent = p.signal_note || '';
  document.getElementById('signal-metrics').innerHTML = `
    <div class="signal-metric"><span class="signal-metric-label">Coverage</span><span class="signal-metric-val">Live DRHP</span></div>
    <div class="signal-metric"><span class="signal-metric-label">Source</span><span class="signal-metric-val">SEBI Filing</span></div>
    <div class="signal-metric"><span class="signal-metric-label">Mode</span><span class="signal-metric-val">LLM Extract</span></div>`;
  document.getElementById('thesis-text').textContent = p.thesis || '—';

  // Metrics
  const fmt = v => v || 'Awaiting data';
  const metrics = [
    {label:'Issue Size',  value: p.issue_size,           note:'From DRHP',       color:'amber'},
    {label:'Price Band',  value: p.price_band,           note:'Per equity share', color:'yellow'},
    {label:'Implied Val', value: p.implied_value,        note:'Post-issue cap',   color:'blue'},
    {label:'Lot Size',    value: p.lot_size,             note:'Min application',  color:'green'},
    {label:'Subscription',value: p.subscription_status, note:'Current status',   color:'amber'},
    {label:'Grey Market', value: p.grey_market,         note:'Informal premium',  color:'yellow'},
  ];
  document.getElementById('metrics-grid').innerHTML = metrics.map(m => `
    <div class="metric-card ${m.color}">
      <span class="metric-label">${m.label}</span>
      <div class="metric-value">${fmt(m.value)}</div>
      <div class="metric-note">${m.note}</div>
    </div>`).join('');

  // Charts
  renderPeerChart(p);
  renderRiskChart(p);

  // Briefing
  document.getElementById('briefing-text').textContent = p.briefing || '—';
  const ul = (items, cls='') => (items||[]).map(i =>
    `<li class="${cls}">${i}</li>`).join('');
  document.getElementById('upside-list').innerHTML = ul(p.upside);
  document.getElementById('watchlist-list').innerHTML = ul(p.watchlist, 'risk');

  // Proceeds
  const proceeds = [
    {bucket:'Capital Expenditure', val: p.proceeds_capex},
    {bucket:'Debt Repayment',      val: p.proceeds_debt},
    {bucket:'General Corporate',   val: p.proceeds_general},
    {bucket:'Other Purposes',      val: p.proceeds_other},
  ].filter(r => r.val);
  document.getElementById('proceeds-body').innerHTML =
    proceeds.length > 0
      ? proceeds.map(r => `<tr><td>${r.bucket}</td><td>${r.val}</td></tr>`).join('')
      : '<tr><td colspan="2" style="color:var(--muted)">Proceeds breakdown not found in filing</td></tr>';

  // Welcome message in chat
  const chatMsgs = document.getElementById('chat-messages');
  chatMsgs.innerHTML = `<div class="msg assistant"><div class="msg-bubble">
    <strong>${p.company} DRHP loaded.</strong><br><br>
    ${p.status === 'extracted'
      ? `Sector: <strong>${p.sector}</strong>. Ask me anything about this filing — financials, risks, management, business model, or use of proceeds.`
      : 'Still extracting data from the DRHP... You can ask questions now and I\'ll answer from the raw filing text.'}
  </div></div>`;
}

function renderPeerChart(p) {
  const ctx = document.getElementById('peer-chart');
  if (peerChart) peerChart.destroy();
  const peers = p.peers || ['Peer 1','Peer 2','Peer 3','Peer 4'];
  const companies = [p.company, ...peers].slice(0,5);
  const evs = [p.peer_ev_sales?.[0]||2.5, ...(p.peer_ev_sales||[]).slice(1,5)];
  const pes  = [p.peer_pe?.[0]||25.0,     ...(p.peer_pe||[]).slice(1,5)];
  while(evs.length < companies.length) evs.push(evs[evs.length-1]||2);
  while(pes.length  < companies.length) pes.push(pes[pes.length-1]||25);
  peerChart = new Chart(ctx, {
    data: {
      labels: companies,
      datasets: [
        { type:'bar', label:'EV/Sales', data: evs,
          backgroundColor: companies.map((_,i)=>i===0?'rgba(255,179,71,.85)':'rgba(90,111,134,.6)'),
          yAxisID:'y' },
        { type:'line', label:'P/E', data: pes,
          borderColor:'#78b7ff', backgroundColor:'rgba(120,183,255,.15)',
          pointBackgroundColor:'#78b7ff', pointRadius:6,
          yAxisID:'y2', tension:.3 },
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{labels:{color:'#c8d1dc',font:{family:'IBM Plex Mono',size:11}}} },
      scales:{
        x:{ticks:{color:'#9aa7b7',font:{size:10}},grid:{color:'rgba(255,179,71,.06)'}},
        y:{ticks:{color:'#9aa7b7'},grid:{color:'rgba(255,179,71,.06)'},title:{display:true,text:'EV/Sales',color:'#ffb347'}},
        y2:{position:'right',ticks:{color:'#78b7ff'},grid:{display:false},title:{display:true,text:'P/E',color:'#78b7ff'}}
      }
    }
  });
}

function renderRiskChart(p) {
  const ctx = document.getElementById('risk-chart');
  if (riskChart) riskChart.destroy();
  const labels = ['Execution','Regulatory','Supply Chain','Demand','Competition'];
  const vals   = [
    p.risk_execution||30, p.risk_regulatory||20, p.risk_supply_chain||15,
    p.risk_demand||20, p.risk_competition||15
  ];
  riskChart = new Chart(ctx, {
    type:'doughnut',
    data:{ labels, datasets:[{
      data: vals,
      backgroundColor:['#ffb347','#ffe082','#78b7ff','#5ad08c','#7d8ea3'],
      borderColor:'rgba(5,7,10,.9)', borderWidth:2
    }]},
    options:{
      responsive:true, maintainAspectRatio:false, cutout:'62%',
      plugins:{
        legend:{position:'right',labels:{color:'#c8d1dc',font:{family:'IBM Plex Mono',size:10},padding:12}},
        tooltip:{callbacks:{label:ctx=>`${ctx.label}: ${ctx.parsed}%`}}
      }
    }
  });
}

// ── Chat ──────────────────────────────────────────────────────────────────
async function sendChat() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg || !currentCompany) return;
  input.value = '';

  addMessage('user', msg);
  const thinking = addMessage('assistant', '⏳ Analysing DRHP...');

  chatHistory.push({role:'user', content:msg});

  try {
    const r = await fetch(`${API}/api/chat`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({company:currentCompany, message:msg, history:chatHistory.slice(-6)})
    });
    const d = await r.json();
    thinking.querySelector('.msg-bubble').textContent = d.answer;
    chatHistory.push({role:'assistant', content:d.answer});
  } catch(e) {
    thinking.querySelector('.msg-bubble').textContent = 'Network error. Please try again.';
  }
}

function addMessage(role, text) {
  const box = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `<div class="msg-bubble">${text}</div>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
}

function clearChat() {
  chatHistory = [];
  document.getElementById('chat-messages').innerHTML =
    `<div class="msg assistant"><div class="msg-bubble">Chat reset. Ask me anything about ${currentCompany||'the selected company'}'s DRHP.</div></div>`;
}

function quickPrompt(text) {
  document.getElementById('chat-input').value = text;
  sendChat();
}

async function triggerRefresh() {
  await fetch(`${API}/api/refresh`, {method:'POST'});
  setTimeout(loadCompanies, 3000);
  alert('Refresh triggered. New DRHPs will appear in the dropdown within a few minutes.');
}

init();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(HTML)


# ══════════════════════════════════════════════════════════════════════════════
# Scheduler
# ══════════════════════════════════════════════════════════════════════════════
scheduler = BackgroundScheduler()
scheduler.add_job(startup_pipeline, "cron", hour=2, minute=0,
                  id="daily_scrape", coalesce=True, max_instances=1)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Starting IPO Crunch on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
