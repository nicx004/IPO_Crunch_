from __future__ import annotations

from typing import Iterable

from openai import OpenAI


def model_candidates(primary: str, fallbacks: Iterable[str]) -> list[str]:
    candidates = [primary]
    for model_name in fallbacks:
        if model_name and model_name not in candidates:
            candidates.append(model_name)
    return candidates


def is_missing_model_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "no endpoints found" in message or "404" in message


def create_chat_completion(client: OpenAI, messages: list[dict], temperature: float, max_tokens: int, model_name: str):
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def chat_with_model_fallback(
    client: OpenAI,
    primary_model: str,
    fallback_models: list[str],
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> tuple[str, str]:
    last_error: Exception | None = None
    for model_name in model_candidates(primary_model, fallback_models):
        try:
            resp = create_chat_completion(client, messages, temperature, max_tokens, model_name)
            content = resp.choices[0].message.content or ""
            return content, model_name
        except Exception as exc:
            last_error = exc
            if not is_missing_model_error(exc):
                break

    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM call failed without raising a specific error")
