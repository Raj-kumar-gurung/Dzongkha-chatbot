"""Chat engine: history trimming, provider selection, fallback, optional translation."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import strings
from .config import Settings, load_settings
from .prompt import SYSTEM_PROMPT
from .providers import Provider, ProviderError, build_providers
from .translate import GoogleTranslator

log = logging.getLogger("dzongkha_bot")

_ENGLISH_REQUEST = re.compile(
    r"\b(in english|answer in english|reply in english|respond in english|english please)\b",
    re.IGNORECASE,
)


@dataclass
class Reply:
    text: str
    provider: Optional[str] = None  # which provider answered (None on failure)
    translated: bool = False
    error: Optional[str] = None  # error kind when no provider answered
    # Translation-pipeline extras (None in direct mode)
    en: Optional[str] = None  # English version of the answer
    user_en: Optional[str] = None  # English version of the user's message
    backend: Optional[str] = None  # translation backend that produced the Dzongkha
    degraded: bool = False  # True when the answer is shown in English because translation failed


class ChatEngine:
    def __init__(
        self,
        settings: Settings,
        providers: List[Provider],
        translator: Optional[GoogleTranslator] = None,
        clock=time.monotonic,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.system_prompt = system_prompt
        self.s = settings
        self.providers = providers
        self.translator = translator
        self._clock = clock
        self._cooldown_until: Dict[str, float] = {}

    # -- history ------------------------------------------------------------ #
    def trim_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Keep only the last N turns and cap the length of older messages, so the
        free quota is not wasted resending a long conversation. The newest user
        message gets a larger cap so a long question is not cut short."""
        keep = messages[-(self.s.max_history_turns * 2 + 1):]
        while keep and keep[0]["role"] != "user":
            keep = keep[1:]
        cap = self.s.max_message_chars
        out = [{"role": m["role"], "content": m["content"][:cap]} for m in keep[:-1]]
        out.append({"role": keep[-1]["role"], "content": keep[-1]["content"][: cap * 4]})
        return out

    # -- main --------------------------------------------------------------- #
    def reply(self, messages: List[Dict[str, str]]) -> Reply:
        """`messages` is the session history INCLUDING the latest user message."""
        if not messages or messages[-1]["role"] != "user":
            return Reply(strings.ERR_GENERIC, error="other")
        trimmed = self.trim_history(messages)
        english_requested = bool(_ENGLISH_REQUEST.search(messages[-1]["content"]))

        kinds: List[str] = []
        for p in self.providers:
            if not p.is_configured():
                kinds.append("not_configured")
                continue
            if self._clock() < self._cooldown_until.get(p.name, 0.0):
                kinds.append("rate_limit")  # recently rate limited: do not waste a call
                continue
            try:
                text = p.generate(self.system_prompt, trimmed)
            except ProviderError as e:
                kinds.append(e.kind)
                log.warning("Provider %s failed (%s: %s)", p.describe(), e.kind, e.detail)
                if e.kind == "rate_limit":
                    self._cooldown_until[p.name] = self._clock() + self.s.rate_limit_cooldown
                continue  # next provider; only reached after a failure

            log.info("Answered by %s", p.describe())
            translated = False
            if self.translator and self.translator.should_translate(text, english_requested):
                out = self.translator.to_dzongkha(text)
                if out:
                    text, translated = out, True
            return Reply(text, provider=p.describe(), translated=translated)

        kind, text = self._error(kinds)
        return Reply(text, error=kind)

    @staticmethod
    def _error(kinds: List[str]):
        """Pick the most useful error kind and its Dzongkha message."""
        if "rate_limit" in kinds:
            return "rate_limit", strings.ERR_RATE_LIMIT
        if all(k == "not_configured" for k in kinds):  # also true for no providers
            return "not_configured", strings.ERR_NOT_CONFIGURED
        if "unavailable" in kinds:
            return "unavailable", strings.ERR_UNAVAILABLE
        return "other", strings.ERR_GENERIC


def build_engine(settings: Optional[Settings] = None):
    """PIPELINE_MODE=nllb (default): English chat model + local translators.
    PIPELINE_MODE=direct: the chat model writes Dzongkha itself."""
    s = settings or load_settings()
    providers = build_providers(s)
    if s.pipeline_mode == "nllb":
        from .pipeline import PipelineEngine
        from .prompt import ENGLISH_PIPELINE_PROMPT
        from .translation_service import build_translation_service

        chat = ChatEngine(s, providers, None, system_prompt=ENGLISH_PIPELINE_PROMPT)
        service = build_translation_service(s)
        log.info(
            "Pipeline mode | chat: %s | translators: %s",
            ", ".join(p.describe() for p in providers) or "none",
            ", ".join(b.describe() for b in service.backends) or "none",
        )
        return PipelineEngine(s, chat, service)
    translator = GoogleTranslator(s) if s.enable_translate else None
    log.info(
        "Direct mode | providers: %s | google translate=%s",
        ", ".join(p.describe() for p in providers) or "none",
        "on" if translator and translator.is_enabled() else "off",
    )
    return ChatEngine(s, providers, translator)
