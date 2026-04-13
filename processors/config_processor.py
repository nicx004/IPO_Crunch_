from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    ipo_dir: Path
    llm_key: str
    llm_model: str
    llm_base: str
    llm_fallback_models: list[str]
    enable_web_enrichment: bool
    port: int
    refresh_token: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        ipo_dir = Path(os.getenv("IPO_DATA_DIR", "./ipo_data"))
        ipo_dir.mkdir(exist_ok=True)

        return cls(
            ipo_dir=ipo_dir,
            llm_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free"),
            llm_base=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            llm_fallback_models=[
                m.strip()
                for m in os.getenv("LLM_MODEL_FALLBACKS", "google/gemma-4-26b-a4b-it:free").split(",")
                if m.strip()
            ],
            enable_web_enrichment=os.getenv("ENABLE_WEB_ENRICHMENT", "1").strip().lower() in {"1", "true", "yes", "on"},
            port=int(os.getenv("PORT", 7860)),
            refresh_token=os.getenv("REFRESH_TOKEN", "").strip(),
        )


def cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]

    render_external = os.getenv("RENDER_EXTERNAL_URL", "").strip()
    if render_external:
        if render_external.startswith("http://") or render_external.startswith("https://"):
            return [render_external]
        return [f"https://{render_external}"]

    return ["*"]


def company_key(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()
