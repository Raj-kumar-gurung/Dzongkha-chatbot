"""Evaluate TRANSLATION quality separately from conversation quality.

    python -m eval.run_translation_eval

Needs the translation models (python download_models.py). Writes
eval/translation_results.md. Round-trip (en->dz->en) similarity is only a rough
signal: a bad translation can round-trip well and a good one can come back
reworded. A fluent Dzongkha speaker must judge the Dzongkha column.
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dzongkha_bot.config import load_settings  # noqa: E402
from dzongkha_bot.textutil import script_ratio  # noqa: E402
from dzongkha_bot.translation_service import TranslationUnavailable, build_translation_service  # noqa: E402
from eval.translation_cases import DZ_SENTENCES, EN_SENTENCES, REFERENCE_DZ  # noqa: E402


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Simple character n-gram F-score (0..1), spaces ignored."""
    h, r = hyp.replace(" ", ""), ref.replace(" ", "")
    ps, rs = [], []
    for k in range(1, n + 1):
        hc = Counter(h[i : i + k] for i in range(len(h) - k + 1))
        rc = Counter(r[i : i + k] for i in range(len(r) - k + 1))
        if not hc or not rc:
            continue
        match = sum((hc & rc).values())
        ps.append(match / sum(hc.values()))
        rs.append(match / sum(rc.values()))
    if not ps:
        return 0.0
    p, rec = sum(ps) / len(ps), sum(rs) / len(rs)
    return 0.0 if p + rec == 0 else (1 + beta**2) * p * rec / (beta**2 * p + rec)


def words_overlap(a: str, b: str) -> float:
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


def main() -> None:
    service = build_translation_service(load_settings())
    rows = []
    for en in EN_SENTENCES:
        try:
            dz, b1 = service.to_dzongkha(en)
            back, b2 = service.to_english(dz)
            ref = REFERENCE_DZ.get(en)
            score = f"{chrf(dz, ref):.2f}" if ref else "n/a (no reference)"
            rows.append(("EN→DZ→EN", en, dz, back, f"{script_ratio(dz):.2f}", f"{words_overlap(en, back):.2f}", score, b1 or "-"))
        except TranslationUnavailable as e:
            rows.append(("EN→DZ→EN", en, "FAILED", str(e), "-", "-", "-", "-"))
        print("done:", en[:50])
    for dz in DZ_SENTENCES:
        try:
            en, b = service.to_english(dz)
            rows.append(("DZ→EN", dz, "", en, "-", "-", "-", b or "-"))
        except TranslationUnavailable as e:
            rows.append(("DZ→EN", dz, "FAILED", str(e), "-", "-", "-", "-"))
        print("done:", dz[:30])

    out = ROOT / "eval" / "translation_results.md"
    lines = [
        "# Translation evaluation (separate from chat quality)\n",
        "Round-trip word overlap is a rough signal only. **Fluent-speaker review is required.**\n",
        "| Direction | Source | Dzongkha output | Back / English output | Tibetan-script share | Round-trip word overlap | chrF vs reference | Backend | Fluent-speaker review |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in r) + " |  |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote", out)


if __name__ == "__main__":
    main()
