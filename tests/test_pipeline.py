"""Offline tests for the translation pipeline using fake backends and a fake
chat model. No network, no torch, no cost. These test the LOGIC (routing,
fallback, error handling), not translation quality."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from dzongkha_bot import strings
from dzongkha_bot.config import load_settings
from dzongkha_bot.engine import ChatEngine
from dzongkha_bot.hf_translator import BackendError, HFBackend
from dzongkha_bot.pipeline import PipelineEngine
from dzongkha_bot.prompt import ENGLISH_PIPELINE_PROMPT
from dzongkha_bot.providers import Provider, ProviderError
from dzongkha_bot.textutil import (
    chunk_long,
    detect_language,
    split_code_blocks,
    translate_text_to_dzongkha,
    translate_text_to_english,
)
from dzongkha_bot.translation_service import TranslationService, TranslationUnavailable

DZ_Q = "ཁྱོད་ཀྱི་མིང་ག་ཅི་ཨིན་ན།"
DZ_A = "བཀྲ་ཤིས་བདེ་ལེགས། ང་ནི་རྫོང་ཁའི་ཁ་སླབ་མཁན་ཅིག་ཨིན།"


def settings(**env):
    return load_settings({k.upper(): str(v) for k, v in env.items()})


class FakeBackend:
    """en->dz returns Dzongkha script; dz->en returns 'EN(<n>)'. Counts calls."""

    def __init__(self, name="fake", pairs=(("en", "dz"), ("dz", "en")), fail=False, bad_script=False):
        self.name, self.pairs, self.fail, self.bad_script = name, set(pairs), fail, bad_script
        self.calls = []

    def supports(self, s, t):
        return (s, t) in self.pairs

    def describe(self):
        return self.name

    def load(self):
        pass

    def translate_batch(self, texts, src, tgt):
        self.calls.append((src, tgt, list(texts)))
        if self.fail:
            raise BackendError("boom")
        if tgt == "dz":
            return [("Latin text" if self.bad_script else DZ_A) for _ in texts]
        return [("དྲི་བ་" if self.bad_script else f"English({i})") for i, _ in enumerate(texts)]


class Chat(Provider):
    name = "chat"

    def __init__(self, answer="My name is Bot.", error=None):
        self.answer, self.error, self.seen = answer, error, []

    def is_configured(self):
        return True

    def generate(self, system, messages):
        self.seen.append((system, [dict(m) for m in messages]))
        if self.error:
            raise self.error
        return self.answer


def make(backends=None, chat=None, **env):
    s = settings(**env)
    backends = backends if backends is not None else [FakeBackend()]
    chat = chat or Chat()
    ce = ChatEngine(s, [chat], None, system_prompt=ENGLISH_PIPELINE_PROMPT)
    return PipelineEngine(s, ce, TranslationService(backends, s.translate_max_sentences)), chat, backends


def user(t, en=None):
    m = {"role": "user", "content": t}
    if en is not None:
        m["en"] = en
    return m


# ------------------------------------------------------------------ flow
def test_english_question_gets_dzongkha_answer_and_input_not_translated():
    eng, chat, (b,) = make()
    r = eng.reply([user("What is your name?")])
    assert r.text == DZ_A and r.translated and r.error is None and r.degraded is False
    assert r.user_en == "What is your name?" and r.en.startswith("My name")
    # only the OUTPUT was translated (en->dz); no dz->en call for English input
    assert [(c[0], c[1]) for c in b.calls] == [("en", "dz")]
    assert chat.seen[0][0] == ENGLISH_PIPELINE_PROMPT
    assert chat.seen[0][1][-1]["content"] == "What is your name?"


def test_dzongkha_question_is_translated_in_and_out_and_chat_never_sees_tibetan():
    eng, chat, (b,) = make()
    r = eng.reply([user(DZ_Q)])
    assert [(c[0], c[1]) for c in b.calls] == [("dz", "en"), ("en", "dz")]
    sent = chat.seen[0][1][-1]["content"]
    assert sent.startswith("English(") and not any("ༀ" <= c <= "࿿" for c in sent)
    assert r.text == DZ_A


def test_mixed_message_translates_only_the_dzongkha_part():
    eng, chat, (b,) = make()
    eng.reply([user("What is machine learning? " + DZ_Q)])
    assert b.calls[0][:2] == ("dz", "en") and b.calls[0][2] == [DZ_Q]
    sent = chat.seen[0][1][-1]["content"]
    assert "What is machine learning?" in sent and "English(" in sent


def test_followup_uses_stored_english_history():
    eng, chat, (b,) = make()
    msgs = [
        user(DZ_Q, en="What is your name?"),
        {"role": "assistant", "content": DZ_A, "en": "My name is Bot."},
        user("And how old are you?"),
    ]
    eng.reply(msgs)
    hist = chat.seen[0][1]
    assert [m["content"] for m in hist] == ["What is your name?", "My name is Bot.", "And how old are you?"]
    assert all(c[:2] != ("dz", "en") for c in b.calls)  # history was NOT re-translated


def test_explicit_english_request_skips_output_translation():
    eng, chat, (b,) = make()
    r = eng.reply([user("Please answer in English: what is a cat?")])
    assert r.text == chat.answer and not r.translated and b.calls == []


# --------------------------------------------------------------- failures
def test_input_translation_failure_gives_dzongkha_error_and_no_chat_call():
    eng, chat, _ = make(backends=[FakeBackend(fail=True)])
    r = eng.reply([user(DZ_Q)])
    assert r.text == strings.ERR_TRANSLATION_UNAVAILABLE and r.error == "translation"
    assert chat.seen == []


def test_output_translation_failure_shows_notice_plus_english_not_a_fake_translation():
    # dz->en works (not needed for English input); en->dz is down
    eng, chat, _ = make(backends=[FakeBackend(pairs=(("dz", "en"),))])
    r = eng.reply([user("Hello there")])
    assert r.degraded and r.error is None and r.translated is False
    assert r.text.startswith(strings.NOTE_TRANSLATION_FAILED) and chat.answer in r.text
    assert r.en == chat.answer  # history keeps clean English


def test_backend_fallback_order_and_script_validation():
    bad, good = FakeBackend("cst", bad_script=True), FakeBackend("nllb-stock")
    eng, _, _ = make(backends=[bad, good])
    r = eng.reply([user("Hi")])
    assert r.text == DZ_A and r.backend == "nllb-stock"
    assert len(bad.calls) == 1 and len(good.calls) == 1  # bad one tried once, then fallback


def test_chat_error_passes_through_in_dzongkha():
    eng, _, _ = make(chat=Chat(error=ProviderError("rate_limit")))
    r = eng.reply([user("Hi")])
    assert r.text == strings.ERR_RATE_LIMIT and r.error == "rate_limit"


def test_chat_down_means_no_output_translation_attempt():
    eng, _, (b,) = make(chat=Chat(error=ProviderError("unavailable")))
    eng.reply([user("Hi")])
    assert b.calls == []


def test_long_message_over_sentence_cap_is_a_clean_error():
    eng, chat, _ = make(TRANSLATE_MAX_SENTENCES=3)
    long_dz = "".join(f"ཁྱོད་ཀྱི་མིང་{i}།" for i in range(10))
    r = eng.reply([user(long_dz)])
    assert r.error == "translation" and chat.seen == []


def test_long_english_answer_is_chunked_not_truncated():
    answer = " ".join(f"This is sentence number {i}." for i in range(12))
    eng, _, (b,) = make(chat=Chat(answer=answer))
    eng.reply([user("Tell me a lot")])
    sent = b.calls[0][2]
    assert len(sent) == 12 and sent[0] == "This is sentence number 0."


# --------------------------------------------------------------- textutil
def test_detect_language():
    assert detect_language("hello") == "en"
    assert detect_language(DZ_Q) == "dz"
    assert detect_language("hello " + DZ_Q) == "mixed"
    assert detect_language("12345") == "en"


def test_code_blocks_and_list_markers_are_preserved():
    text = "Here is code:\n```python\nprint('hi')\n```\n- First item.\n- Second item.\n\n1) Third."
    seen = []

    def fake(batch):
        seen.extend(batch)
        return [f"<{s}>" for s in batch]

    out = translate_text_to_dzongkha(text, fake, 50)
    assert "```python\nprint('hi')\n```" in out
    assert "- <First item.>" in out and "- <Second item.>" in out and "1) <Third.>" in out
    assert "print('hi')" not in " ".join(seen)  # code never sent to the translator
    assert "\n\n" in out  # blank line kept


def test_numbers_only_lines_are_not_translated():
    seen = []
    out = translate_text_to_dzongkha("42\nHello.", lambda b: (seen.extend(b) or [f"<{x}>" for x in b]), 10)
    assert seen == ["Hello."] and out.startswith("42")


def test_to_english_leaves_english_parts_and_numbers_alone():
    out = translate_text_to_english("In 2024 " + DZ_Q + " ok?", lambda b: ["WHAT"], 10)
    assert out == "In 2024 WHAT ok?"


def test_chunk_long_sentence():
    s = ", ".join(["word " * 10] * 10)
    parts = chunk_long(s, 120)
    assert len(parts) > 1 and all(len(p) <= 180 for p in parts)


def test_service_to_english_pure_english_makes_no_call():
    b = FakeBackend()
    out, used = TranslationService([b]).to_english("just english")
    assert out == "just english" and used is None and b.calls == []


def test_service_raises_when_no_backend_supports_direction():
    with pytest.raises(TranslationUnavailable):
        TranslationService([FakeBackend(pairs=(("dz", "en"),))]).to_dzongkha("Hello")


# --------------------------------------------------------------- config
def test_pipeline_is_default_and_local_only():
    s = settings()
    assert s.pipeline_mode == "nllb" and s.provider_order == ("ollama",)
    assert s.translator_order == ("cst", "nllb") and s.translate_device == "cpu"
    assert s.ollama_model == "llama3.2:3b"
    assert settings(PIPELINE_MODE="direct").pipeline_mode == "direct"
    assert settings(PIPELINE_MODE="nonsense").pipeline_mode == "nllb"


def test_hf_backend_reports_missing_packages_cleanly():
    try:
        import torch  # noqa: F401
        pytest.skip("torch is installed; this test only covers the missing-package message")
    except ImportError:
        pass
    b = HFBackend("x", "some/model", {("en", "dz")})
    with pytest.raises(BackendError, match="not installed"):
        b.load()
