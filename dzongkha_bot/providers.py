"""AI providers. Each one turns (system prompt, messages) into a reply string.

Failures are normalised into ProviderError so the engine can decide whether to
fall back. A provider is only ever asked once per user message.
"""
from __future__ import annotations

import logging
from typing import List, Dict

import requests

from .config import Settings

log = logging.getLogger("dzongkha_bot")

Message = Dict[str, str]  # {"role": "user" | "assistant", "content": "..."}


class ProviderError(Exception):
    """kind: not_configured | rate_limit | unavailable | auth | bad_model | other"""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail


class Provider:
    name = "base"

    def is_configured(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> str:
        return self.name

    def generate(self, system: str, messages: List[Message]) -> str:  # pragma: no cover
        raise NotImplementedError


def _kind_from_status(status: int, body: str = "") -> str:
    if status == 429:
        return "rate_limit"
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "bad_model"
    if status == 400 and "api key" in body.lower():
        return "auth"
    if status >= 500:
        return "unavailable"
    return "other"


# --------------------------------------------------------------------------- #
# Google Gemini (official Google Gen AI SDK)
# --------------------------------------------------------------------------- #
class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, s: Settings):
        self.s = s
        self._client = None

    def is_configured(self) -> bool:
        return bool(self.s.gemini_api_key)

    def describe(self) -> str:
        return f"gemini:{self.s.gemini_model}"

    def _get_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self.s.gemini_api_key,
                http_options=types.HttpOptions(timeout=self.s.request_timeout * 1000),
            )
        return self._client

    def generate(self, system: str, messages: List[Message]) -> str:
        if not self.is_configured():
            raise ProviderError("not_configured", "GEMINI_API_KEY is empty")
        from google.genai import errors, types

        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        def call(use_search: bool):
            return self._get_client().models.generate_content(
                model=self.s.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=self.s.temperature,
                    max_output_tokens=self.s.max_output_tokens,
                    # Google Search grounding lets the model check current and
                    # specific facts instead of guessing.
                    tools=[types.Tool(google_search=types.GoogleSearch())] if use_search else None,
                ),
            )

        use_search = self.s.gemini_use_search and not getattr(self, "s_search_disabled", False)
        try:
            try:
                resp = call(use_search)
            except errors.APIError as e:
                code = int(getattr(e, "code", 0) or 0)
                # Search tool not allowed for this model/key (400/403): retry once
                # without it. Never retry rate limits or outages (wastes quota).
                if use_search and code in (400, 403):
                    log.warning("Gemini search grounding refused (HTTP %s); retrying without it", code)
                    self.s_search_disabled = True
                    resp = call(False)
                else:
                    raise
        except errors.APIError as e:
            code = int(getattr(e, "code", 0) or 0)
            raise ProviderError(_kind_from_status(code, str(e)), f"HTTP {code}") from None
        except Exception as e:  # network errors, timeouts
            raise ProviderError("unavailable", type(e).__name__) from None

        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            raise ProviderError("other", "empty or blocked response")
        return text


# --------------------------------------------------------------------------- #
# Any OpenAI-compatible endpoint (Groq, OpenRouter, ...) - optional
# --------------------------------------------------------------------------- #
class OpenAICompatProvider(Provider):
    name = "openai_compat"

    def __init__(self, s: Settings):
        self.s = s

    def is_configured(self) -> bool:
        if not (self.s.compat_base_url and self.s.compat_api_key and self.s.compat_model):
            return False
        # Cost guard: on OpenRouter only ":free" models cost nothing.
        if "openrouter.ai" in self.s.compat_base_url and not self.s.compat_model.endswith(":free"):
            log.warning(
                "Refusing OpenRouter model %r: only ':free' models are allowed here "
                "so this app can never spend credits.",
                self.s.compat_model,
            )
            return False
        return True

    def describe(self) -> str:
        return f"openai_compat:{self.s.compat_model}"

    def generate(self, system: str, messages: List[Message]) -> str:
        if not self.is_configured():
            raise ProviderError("not_configured", "OPENAI_COMPAT_* not fully set or model not free")
        payload = {
            "model": self.s.compat_model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": self.s.temperature,
            "max_tokens": self.s.max_output_tokens,
        }
        try:
            r = requests.post(
                f"{self.s.compat_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.s.compat_api_key}"},
                json=payload,
                timeout=self.s.request_timeout,
            )
        except requests.RequestException as e:
            raise ProviderError("unavailable", type(e).__name__) from None
        if r.status_code != 200:
            raise ProviderError(_kind_from_status(r.status_code, r.text), f"HTTP {r.status_code}")
        try:
            text = (r.json()["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, ValueError):
            raise ProviderError("other", "unexpected response shape") from None
        if not text:
            raise ProviderError("other", "empty response")
        return text


# --------------------------------------------------------------------------- #
# Local Ollama (no API key, no cloud, no cost)
# --------------------------------------------------------------------------- #
class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, s: Settings):
        self.s = s

    def is_configured(self) -> bool:
        return bool(self.s.ollama_host and self.s.ollama_model)

    def describe(self) -> str:
        return f"ollama:{self.s.ollama_model}"

    def generate(self, system: str, messages: List[Message]) -> str:
        payload = {
            "model": self.s.ollama_model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "options": {
                "temperature": self.s.temperature,
                "num_predict": self.s.max_output_tokens,
            },
        }
        try:
            # Local models can be slow on first load, so allow a longer timeout.
            r = requests.post(
                f"{self.s.ollama_host}/api/chat",
                json=payload,
                timeout=max(self.s.request_timeout, 300),
            )
        except requests.ConnectionError:
            raise ProviderError("unavailable", "Ollama is not running") from None
        except requests.RequestException as e:
            raise ProviderError("unavailable", type(e).__name__) from None
        if r.status_code == 404:
            raise ProviderError("bad_model", f"model not pulled: {self.s.ollama_model}")
        if r.status_code != 200:
            raise ProviderError(_kind_from_status(r.status_code, r.text), f"HTTP {r.status_code}")
        try:
            text = (r.json()["message"]["content"] or "").strip()
        except (KeyError, ValueError):
            raise ProviderError("other", "unexpected response shape") from None
        if not text:
            raise ProviderError("other", "empty response")
        return text


def build_providers(s: Settings) -> List[Provider]:
    registry = {
        "gemini": GeminiProvider,
        "openai_compat": OpenAICompatProvider,
        "ollama": OllamaProvider,
    }
    return [registry[name](s) for name in s.provider_order]
