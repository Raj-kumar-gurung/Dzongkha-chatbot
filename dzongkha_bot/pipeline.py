"""Translation pipeline: Dzongkha/English in -> English chat model -> Dzongkha out.

1. Detect the language of the user's message (no model call).
2. Translate Dzongkha (or the Dzongkha parts of a mixed message) to English.
   Pure English is passed through untouched.
3. Answer with the local chat model (Ollama llama3.2:3b by default) in English,
   using the English version of the history.
4. Translate the English answer to Dzongkha, unless the user explicitly asked
   for English.

The chat model never sees Dzongkha script, and a Dzongkha answer is never
translated twice. If translation fails, the user is told in Dzongkha; an
English answer is shown only with a notice that it could not be translated.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from . import strings
from .engine import _ENGLISH_REQUEST, ChatEngine, Reply
from .textutil import has_tibetan
from .translation_service import TranslationService, TranslationUnavailable

log = logging.getLogger("dzongkha_bot")


class PipelineEngine:
    def __init__(self, settings, chat: ChatEngine, translator: TranslationService):
        self.s = settings
        self.chat = chat
        self.tr = translator

    def warm_up(self) -> None:
        self.tr.warm_up()

    def _english_history(self, earlier: List[Dict[str, str]]) -> List[Dict[str, str]]:
        out = []
        for m in earlier:
            en = m.get("en")
            if en is None:  # message saved without an English version
                en = m["content"]
                if has_tibetan(en):
                    try:
                        en, _ = self.tr.to_english(en)
                    except TranslationUnavailable:
                        continue  # skip rather than send Tibetan script to the chat model
            out.append({"role": m["role"], "content": en})
        return out

    def reply(self, messages: List[Dict[str, str]]) -> Reply:
        """`messages`: session history incl. the latest user message. Each item
        has "content" (as displayed) and optionally "en" (English version)."""
        if not messages or messages[-1]["role"] != "user":
            return Reply(strings.ERR_GENERIC, error="other")
        original = messages[-1]["content"]

        try:
            user_en, in_backend = self.tr.to_english(original)
        except TranslationUnavailable as e:
            log.warning("Input translation failed: %s", e)
            return Reply(strings.ERR_TRANSLATION_UNAVAILABLE, error="translation")

        history = self._english_history(messages[:-1])
        history.append({"role": "user", "content": user_en})
        chat = self.chat.reply(history)
        if chat.error:
            return Reply(chat.text, error=chat.error, user_en=user_en)

        answer_en = chat.text
        base = dict(provider=chat.provider, en=answer_en, user_en=user_en)

        if _ENGLISH_REQUEST.search(original) or _ENGLISH_REQUEST.search(user_en):
            return Reply(answer_en, **base)  # explicit English request: no translation

        try:
            answer_dz, out_backend = self.tr.to_dzongkha(answer_en)
        except TranslationUnavailable as e:
            log.warning("Output translation failed: %s", e)
            return Reply(
                strings.NOTE_TRANSLATION_FAILED + "\n\n" + answer_en,
                degraded=True,
                **base,
            )
        return Reply(answer_dz, translated=True, backend=out_backend, **base)
