"""Optional Google Cloud Translation (Basic v2 REST API) for English -> Dzongkha.

OFF by default. It is a translation component only, never the reasoning engine.
Cloud Translation needs a billing-enabled Google Cloud project, so this module
keeps a local monthly character counter and refuses to send text once the
configured budget is reached. The counter is a safety net, not Google's own
billing meter.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from .config import ROOT, Settings

log = logging.getLogger("dzongkha_bot")

_ENDPOINT = "https://translation.googleapis.com/language/translate/v2"
_TIBETAN_PUNCT = set("་།༎༏༐༑༔")


def dzongkha_script_ratio(text: str) -> float:
    """Share of letters that are Tibetan-script (Dzongkha and Tibetan share this script).

    Returns 1.0 when there are no letters at all (e.g. only digits), meaning
    "nothing worth translating". This is a script check only: it cannot tell
    Dzongkha from Tibetan.
    """
    tib = sum(1 for c in text if "ༀ" <= c <= "࿿" and c not in _TIBETAN_PUNCT)
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    total = tib + lat
    return 1.0 if total == 0 else tib / total


class GoogleTranslator:
    def __init__(self, s: Settings, usage_file: Optional[Path] = None):
        self.s = s
        self.usage_file = usage_file or (ROOT / ".translate_usage.json")

    def is_enabled(self) -> bool:
        return bool(self.s.enable_translate and self.s.translate_api_key)

    # -- local budget ------------------------------------------------------- #
    def _month(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _used(self) -> int:
        try:
            data = json.loads(self.usage_file.read_text(encoding="utf-8"))
            return int(data["chars"]) if data.get("month") == self._month() else 0
        except (OSError, ValueError, KeyError):
            return 0

    def _record(self, n: int) -> None:
        try:
            self.usage_file.write_text(
                json.dumps({"month": self._month(), "chars": self._used() + n}),
                encoding="utf-8",
            )
        except OSError:
            pass

    # -- public ------------------------------------------------------------- #
    def should_translate(self, answer: str, english_requested: bool) -> bool:
        """Translate only when it can help: enabled, not an English request, and
        the answer is mostly NOT Tibetan-script already (so a good Dzongkha
        answer is never translated twice or degraded)."""
        if not self.is_enabled() or english_requested or not answer.strip():
            return False
        return dzongkha_script_ratio(answer) < self.s.translate_threshold

    def to_dzongkha(self, text: str) -> Optional[str]:
        """Return Dzongkha text, or None on any failure (caller keeps original)."""
        if not self.is_enabled():
            return None
        if self._used() + len(text) > self.s.translate_monthly_char_budget:
            log.warning("Translate skipped: local monthly character budget reached")
            return None
        try:
            r = requests.post(
                _ENDPOINT,
                params={"key": self.s.translate_api_key},
                json={"q": [text], "source": "en", "target": "dz", "format": "text"},
                timeout=20,
            )
        except requests.RequestException as e:
            # Do not log the exception text: it can contain the URL with the key.
            log.warning("Translate request failed: %s", type(e).__name__)
            return None
        if r.status_code != 200:
            log.warning("Translate failed with HTTP %s", r.status_code)
            return None
        try:
            out = r.json()["data"]["translations"][0]["translatedText"].strip()
        except (KeyError, IndexError, ValueError):
            return None
        self._record(len(text))
        return out or None
