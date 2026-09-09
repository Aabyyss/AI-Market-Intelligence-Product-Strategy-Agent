"""One interface over any OpenAI-compatible chat API (Phase 3).

Providers:
  - "openai": the OpenAI API — needs OPENAI_API_KEY in the environment
  - "ollama": local models via Ollama — no key; it exposes the same
    OpenAI-compatible /v1 endpoint on localhost
  - "custom": any other OpenAI-compatible server (set OPENAI_BASE_URL)

Resolution when provider is "auto": use OpenAI if OPENAI_API_KEY is
set, otherwise Ollama if it is reachable on localhost:11434.

Keeping credentials out of the code and in the environment is the
first habit of Phase 6 (production: env vars, no secrets in git).
"""

import os
import socket

import openai

from market_intel import config


class LLMError(RuntimeError):
    """Raised when no provider is available or a call fails."""


def _reachable(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_provider() -> str:
    """Pick a provider: explicit choice, else auto-detect."""
    chosen = (config.LLM_PROVIDER or "auto").lower()
    if chosen != "auto":
        return chosen
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if _reachable("127.0.0.1", 11434):
        return "ollama"
    raise LLMError(
        "no LLM provider found. Options:\n"
        "  1. Ollama (local, free): install from https://ollama.com, then\n"
        "     `ollama pull llama3.2` and start the app\n"
        "  2. OpenAI: set the OPENAI_API_KEY environment variable\n"
        "Or force one:  LLM_PROVIDER=openai|ollama python run_ask.py ..."
    )


def _client(provider: str):
    """An OpenAI SDK client pointed at the chosen provider."""
    if provider == "ollama":
        # The SDK needs an api_key; Ollama ignores its value.
        return openai.OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("provider=openai but OPENAI_API_KEY is not set")
        return openai.OpenAI(api_key=key)
    if provider == "custom":
        if not config.CUSTOM_BASE_URL:
            raise LLMError("provider=custom but OPENAI_BASE_URL is not set")
        return openai.OpenAI(
            base_url=config.CUSTOM_BASE_URL,
            api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        )
    raise LLMError(f"unknown provider: {provider!r}")


def model_for(provider: str) -> str:
    return config.LLM_MODEL or config.LLM_DEFAULT_MODELS.get(
        provider, config.LLM_DEFAULT_MODELS["ollama"]
    )


def chat(
    messages: list[dict],
    provider: str | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    """One chat completion; returns the assistant's reply text.

    ``max_tokens`` overrides config.LLM_MAX_TOKENS per call — agents
    that write longer outputs (report sections) can opt for more room
    without changing the default for quick Q&A.

    ``json_mode=True`` asks the provider for a JSON object reply
    (Ollama's ``format=json``, OpenAI's ``response_format``). It steers
    the model but does not guarantee valid JSON — callers still parse
    and validate the reply (agents.py) and fall back when it is not
    usable.
    """
    provider = resolve_provider() if provider is None else provider
    model = model_for(provider)
    client = _client(provider)
    kwargs: dict = {}
    if json_mode:
        if provider == "ollama":
            # Ollama's OpenAI-compatible layer accepts format=json to
            # force a JSON reply (no markdown fences).
            kwargs["extra_body"] = {"format": "json"}
        elif provider == "openai":
            kwargs["response_format"] = {"type": "json_object"}
        # custom: the server may not support either knob, so ask for
        # JSON in the prompt only.
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=max_tokens or config.LLM_MAX_TOKENS,
            **kwargs,
        )
    except openai.OpenAIError as exc:
        raise LLMError(f"LLM call failed ({provider}/{model}): {exc}") from exc
    return response.choices[0].message.content or ""