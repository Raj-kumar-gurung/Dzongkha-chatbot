"""Text helpers for the translation pipeline: script detection, sentence
splitting, and structure-preserving translation of multi-line text.

Code blocks are never sent to a translator. List markers and line breaks are
kept. Dzongkha sentences end with the shad (།).
"""
from __future__ import annotations

import re
from typing import Callable, List, Tuple

_TIB = "ༀ-࿿"
_TIBETAN_PUNCT = set("་།༎༏༐༑༔")

_CODE_FENCE = re.compile(r"(```.*?```)", re.DOTALL)
_TIB_RUN = re.compile(rf"([{_TIB}]+(?:[ \t]+[{_TIB}]+)*)")
_LIST_MARK = re.compile(r"^(\s*(?:[-*•]|\d{1,2}[.)])\s+)")
_SENT_EN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_SENT_DZ = re.compile(r"(?<=།)\s*")


def tibetan_letters(text: str) -> int:
    return sum(1 for c in text if "ༀ" <= c <= "࿿" and c not in _TIBETAN_PUNCT)


def latin_letters(text: str) -> int:
    return sum(1 for c in text if c.isascii() and c.isalpha())


def has_tibetan(text: str) -> bool:
    return tibetan_letters(text) > 0


def script_ratio(text: str) -> float:
    """Share of letters in Tibetan script (Dzongkha and Tibetan share it).
    1.0 when there are no letters at all."""
    t, l = tibetan_letters(text), latin_letters(text)
    return 1.0 if t + l == 0 else t / (t + l)


def detect_language(text: str) -> str:
    """'dz' (all Tibetan script), 'en' (no Tibetan script), or 'mixed'."""
    t, l = tibetan_letters(text), latin_letters(text)
    if t == 0:
        return "en"
    if l == 0:
        return "dz"
    return "mixed"


def split_code_blocks(text: str) -> List[Tuple[bool, str]]:
    """[(is_code, chunk)] in order. Fenced ``` blocks are code."""
    out = []
    for i, part in enumerate(_CODE_FENCE.split(text)):
        if part:
            out.append((i % 2 == 1, part))
    return out


def chunk_long(s: str, max_chars: int) -> List[str]:
    """Break an over-long sentence at commas/semicolons, then at spaces."""
    if len(s) <= max_chars:
        return [s]
    pieces, cur = [], ""
    for piece in re.split(r"(?<=[,;:])\s+", s):
        if cur and len(cur) + 1 + len(piece) > max_chars:
            pieces.append(cur)
            cur = piece
        else:
            cur = (cur + " " + piece).strip()
    if cur:
        pieces.append(cur)
    final = []
    for c in pieces:
        while len(c) > int(max_chars * 1.5):
            cut = c.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            final.append(c[:cut])
            c = c[cut:].lstrip()
        if c:
            final.append(c)
    return final


def split_sentences(text: str, lang: str, max_chars: int = 250) -> List[str]:
    body = text.strip()
    if not body:
        return []
    raw = _SENT_DZ.split(body) if lang == "dz" else _SENT_EN.split(body)
    out: List[str] = []
    for s in raw:
        s = s.strip()
        if s:
            out.extend(chunk_long(s, max_chars))
    return out


BatchFn = Callable[[List[str]], List[str]]


def translate_text_to_dzongkha(text: str, batch_fn: BatchFn, max_sentences: int, max_chars: int = 250) -> str:
    """Translate English prose to Dzongkha, keeping code blocks, blank lines and
    list markers. `batch_fn` translates a list of English sentences. Raises
    ValueError if the text has more than `max_sentences` sentences."""
    plan = []  # per segment: ("code", text) | ("prose", [(prefix, [job_idx...]) per line])
    jobs: List[str] = []
    for is_code, chunk in split_code_blocks(text):
        if is_code:
            plan.append(("code", chunk))
            continue
        lines = []
        for line in chunk.split("\n"):
            if not line.strip():
                lines.append(("", None))
                continue
            m = _LIST_MARK.match(line)
            prefix = m.group(1) if m else ""
            body = line[len(prefix):] if m else line
            idxs = []
            for s in split_sentences(body, "en", max_chars):
                if latin_letters(s) == 0:  # numbers/symbols only: keep as is
                    idxs.append(("raw", s))
                else:
                    idxs.append(("job", len(jobs)))
                    jobs.append(s)
            lines.append((prefix, idxs))
        plan.append(("prose", lines))
    if len(jobs) > max_sentences:
        raise ValueError(f"too many sentences ({len(jobs)} > {max_sentences})")
    done = batch_fn(jobs) if jobs else []
    if len(done) != len(jobs):
        raise ValueError("translator returned a different number of sentences")

    out_parts = []
    for kind, payload in plan:
        if kind == "code":
            out_parts.append(payload)
            continue
        rendered = []
        for prefix, idxs in payload:
            if idxs is None:
                rendered.append("")
                continue
            words = [done[i] if k == "job" else i for k, i in idxs]
            rendered.append(prefix + " ".join(w.strip() for w in words))
        out_parts.append("\n".join(rendered))
    return "".join(out_parts)


def translate_text_to_english(text: str, batch_fn: BatchFn, max_sentences: int) -> str:
    """Translate only the Tibetan-script parts of `text` to English; English
    parts, numbers and code blocks stay untouched (so mixed messages work and a
    pure-English message needs no translation at all)."""
    plan = []  # ("keep", str) | ("tr", [job_idx...])
    jobs: List[str] = []
    for is_code, chunk in split_code_blocks(text):
        if is_code:
            plan.append(("keep", chunk))
            continue
        for i, piece in enumerate(_TIB_RUN.split(chunk)):
            if not piece:
                continue
            if i % 2 == 0:
                plan.append(("keep", piece))
            else:
                idxs = []
                for s in split_sentences(piece, "dz"):
                    idxs.append(len(jobs))
                    jobs.append(s)
                plan.append(("tr", idxs))
    if len(jobs) > max_sentences:
        raise ValueError(f"too many sentences ({len(jobs)} > {max_sentences})")
    done = batch_fn(jobs) if jobs else []
    if len(done) != len(jobs):
        raise ValueError("translator returned a different number of sentences")

    out = ""
    prev_translated = False
    for kind, payload in plan:
        if kind == "keep":
            piece = payload
            if prev_translated and piece and not piece[0].isspace() and piece[0] not in ".,;:!?)":
                piece = " " + piece
            prev_translated = False
        else:
            piece = " ".join(done[i].strip() for i in payload)
            if out and not out[-1].isspace():
                piece = " " + piece
            prev_translated = True
        out += piece
    return out
