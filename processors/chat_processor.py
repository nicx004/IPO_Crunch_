from __future__ import annotations

from openai import OpenAI

from .llm_processor import chat_with_model_fallback
from .pdf_processor import find_pdf, read_pdf


def chat_with_drhp(
    client: OpenAI,
    llm_key: str,
    primary_model: str,
    fallback_models: list[str],
    ipo_dir,
    company: str,
    message: str,
    history: list[dict],
) -> str:
    if not llm_key:
        return "No API key configured. Set OPENROUTER_API_KEY."

    pdf = find_pdf(ipo_dir, company)
    if not pdf:
        return f"No DRHP file found for {company}. Please try again in a minute."

    context = read_pdf(pdf, max_chars=10000)
    system = f"""You are an expert IPO analyst for {company}.
Use DRHP context, stay concise, and distinguish filing facts from inference.

DRHP Context:
{context}
"""

    messages = [{"role": "system", "content": system}]
    for item in history[-6:]:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": message})

    try:
        content, model_used = chat_with_model_fallback(
            client=client,
            primary_model=primary_model,
            fallback_models=fallback_models,
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
        )
        return content or f"No response generated from {model_used}."
    except Exception as exc:
        return f"Error: {exc}"
