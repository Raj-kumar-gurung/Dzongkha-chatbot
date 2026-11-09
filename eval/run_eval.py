"""Run the 26 test cases against your configured provider(s) and write eval/results.md.

    python -m eval.run_eval            (from the project folder)

Uses about 30 chat calls (plus local translation). It waits between calls to stay under free-tier
per-minute limits (SLEEP_SECONDS). Automatic checks here are only a rough
screen (script ratio, Latin-letter leakage, provider errors). They CANNOT judge
whether the Dzongkha is correct or natural: a fluent Dzongkha speaker must read
the Response column and fill in the review column.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dzongkha_bot.engine import build_engine  # noqa: E402
from dzongkha_bot.translate import dzongkha_script_ratio  # noqa: E402
from eval.test_messages import CASES  # noqa: E402

SLEEP_SECONDS = 0  # local pipeline needs no pause; set ~7 if you add a free API (rate limits)


def main() -> None:
    engine = build_engine()
    rows = []
    for case_id, category, turns in CASES:
        history = []
        reply = None
        for turn in turns:
            history.append({"role": "user", "content": turn})
            reply = engine.reply(history)
            if reply.user_en is not None:
                history[-1]["en"] = reply.user_en  # pipeline mode keeps English versions
            if reply.error:
                break
            history.append({"role": "assistant", "content": reply.text, "en": reply.en})
            time.sleep(SLEEP_SECONDS)
        ratio = dzongkha_script_ratio(reply.text)
        expects_english = case_id == "B2"
        if reply.error:
            auto = f"PROVIDER ERROR ({reply.error})"
        elif expects_english:
            auto = "ok (English expected)" if ratio < 0.2 else "CHECK: expected English"
        elif ratio < 0.6:
            auto = "CHECK: not mostly Tibetan script"
        else:
            auto = "script ok"
        if reply.degraded:
            auto = "TRANSLATION FAILED (English shown)"
        route = (reply.provider or "-") + (f" + {reply.backend}" if reply.backend else "")
        shown = reply.text + (f"  [English: {reply.en}]" if reply.en and not reply.error else "")
        rows.append((case_id, category, " ⏎ ".join(turns)[:200], route, f"{ratio:.2f}", auto, shown))
        print(f"{case_id}: {auto}")

    out = ROOT / "eval" / "results.md"
    lines = [
        "# Dzongkha chatbot evaluation\n",
        "Automatic checks are rough. **Fluent-speaker review is required** for correctness and naturalness.\n",
        "| ID | Category | Input | Chat model + translator | Script ratio | Auto check | Response | Fluent-speaker review |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cells = [c.replace("|", "\\|").replace("\n", " ") for c in r]
        lines.append("| " + " | ".join(cells) + " |  |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
