"""Translation service: tries backends in order, validates output script, and
reports failure clearly instead of returning garbage.

There is deliberately NO GovTech backend: GovTech has not published a public
API or weights (see README). Add one here only if GovTech documents one.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from .config import Settings
from .hf_translator import BackendError, HFBackend
from .textutil import (
    detect_language,
    latin_letters,
    script_ratio,
    translate_text_to_dzongkha,
    translate_text_to_english,
)

log = logging.getLogger("dzongkha_bot")


class TranslationUnavailable(Exception):
    """No backend could translate this text."""


class TranslationService:
    def __init__(self, backends: list, max_sentences: int = 40):
        self.backends = backends
        self.max_sentences = max_sentences

    # -- core --------------------------------------------------------------- #
    def _run(self, sentences: List[str], src: str, tgt: str) -> Tuple[List[str], str]:
        """Translate sentences with the first backend that works AND returns the
        expected script. Local backends only: nothing here can cost money."""
        tried = 0
        for b in self.backends:
            if not b.supports(src, tgt):
                continue
            tried += 1
            try:
                out = b.translate_batch(sentences, src, tgt)
            except BackendError as e:
                log.warning("Translation backend %s failed: %s", b.describe(), e)
                continue
            joined = " ".join(out)
            ratio = script_ratio(joined)
            if tgt == "dz" and ratio < 0.6:
                log.warning("Backend %s did not return Dzongkha script (ratio %.2f)", b.describe(), ratio)
                continue
            if tgt == "en" and ratio > 0.2:
                log.warning("Backend %s did not return English (Tibetan ratio %.2f)", b.describe(), ratio)
                continue
            return out, b.name
        raise TranslationUnavailable(f"no working backend for {src}->{tgt} (tried {tried})")

    # -- public ------------------------------------------------------------- #
    def to_english(self, text: str) -> Tuple[str, Optional[str]]:
        """English text is returned untouched (no model call). Dzongkha, or the
        Dzongkha parts of a mixed message, are translated. -> (text, backend|None)"""
        if detect_language(text) == "en":
            return text, None
        used: List[str] = []

        def batch(sents):
            out, name = self._run(sents, "dz", "en")
            used.append(name)
            return out

        try:
            return translate_text_to_english(text, batch, self.max_sentences), (used[0] if used else None)
        except ValueError as e:
            raise TranslationUnavailable(str(e)) from None

    def to_dzongkha(self, text: str) -> Tuple[str, Optional[str]]:
        """-> (dzongkha text, backend). Text with no letters is returned as is."""
        if latin_letters(text) == 0:
            return text, None
        used: List[str] = []

        def batch(sents):
            out, name = self._run(sents, "en", "dz")
            used.append(name)
            return out

        try:
            return translate_text_to_dzongkha(text, batch, self.max_sentences), (used[0] if used else None)
        except ValueError as e:
            raise TranslationUnavailable(str(e)) from None

    def warm_up(self) -> None:
        """Load the English->Dzongkha model now so the first answer is not slow."""
        for b in self.backends:
            if b.supports("en", "dz"):
                try:
                    b.load()
                    return
                except BackendError as e:
                    log.warning("Warm-up failed for %s: %s", b.describe(), e)


def build_translation_service(s: Settings) -> TranslationService:
    backends = []
    common = dict(device=s.translate_device, local_files_only=s.translate_local_files_only, beams=s.translate_beams)
    for name in s.translator_order:
        if name == "cst":
            backends.append(HFBackend("cst-en-dz", s.cst_en_to_dz_model, {("en", "dz")}, **common))
            backends.append(HFBackend("cst-dz-en", s.cst_dz_to_en_model, {("dz", "en")}, **common))
        elif name == "nllb":
            backends.append(HFBackend("nllb-stock", s.nllb_stock_model, {("en", "dz"), ("dz", "en")}, **common))
    return TranslationService(backends, max_sentences=s.translate_max_sentences)
