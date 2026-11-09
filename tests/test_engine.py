"""Offline tests with fake providers. No network, no API key, no cost."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from dzongkha_bot import strings
from dzongkha_bot.config import load_settings
from dzongkha_bot.engine import ChatEngine
from dzongkha_bot.prompt import SYSTEM_PROMPT
from dzongkha_bot.providers import (
    GeminiProvider,
    OpenAICompatProvider,
    Provider,
    ProviderError,
)
from dzongkha_bot.translate import GoogleTranslator, dzongkha_script_ratio

DZ = "བཀྲ་ཤིས་བདེ་ལེགས། ང་ནི་རྫོང་ཁའི་ཁ་སླབ་མཁན་ཅིག་ཨིན།"


class Fake(Provider):
    def __init__(self, name, result=DZ, configured=True):
        self.name, self.result, self._cfg = name, result, configured
        self.calls, self.last_system, self.last_messages = 0, None, None

    def is_configured(self):
        return self._cfg

    def generate(self, system, messages):
        self.calls += 1
        self.last_system, self.last_messages = system, messages
        if isinstance(self.result, ProviderError):
            raise self.result
        return self.result


def settings(**env):
    return load_settings({k.upper(): str(v) for k, v in env.items()})


def user(t):
    return {"role": "user", "content": t}


def test_first_provider_answers_and_second_is_not_called():
    a, b = Fake("a"), Fake("b")
    r = ChatEngine(settings(), [a, b]).reply([user("hi")])
    assert r.text == DZ and r.provider == "a" and r.error is None
    assert (a.calls, b.calls) == (1, 0)  # never sends the same question twice


def test_fallback_after_failure():
    a, b = Fake("a", ProviderError("unavailable")), Fake("b")
    r = ChatEngine(settings(), [a, b]).reply([user("hi")])
    assert r.provider == "b" and (a.calls, b.calls) == (1, 1)


def test_rate_limit_gives_dzongkha_message_and_cooldown():
    t = [0.0]
    a = Fake("a", ProviderError("rate_limit"))
    e = ChatEngine(settings(), [a], clock=lambda: t[0])
    r = e.reply([user("hi")])
    assert r.text == strings.ERR_RATE_LIMIT and r.error == "rate_limit"
    e.reply([user("hi again")])
    assert a.calls == 1  # cooldown: no wasted second call
    t[0] += 61
    e.reply([user("hi once more")])
    assert a.calls == 2


def test_unconfigured_everything():
    r = ChatEngine(settings(), [Fake("a", configured=False)]).reply([user("hi")])
    assert r.text == strings.ERR_NOT_CONFIGURED and r.error is not None


def test_no_providers_does_not_crash():
    assert ChatEngine(settings(), []).reply([user("hi")]).error is not None


def test_system_prompt_sent_every_time():
    a = Fake("a")
    e = ChatEngine(settings(), [a])
    e.reply([user("one")])
    e.reply([user("one"), {"role": "assistant", "content": DZ}, user("two")])
    assert a.last_system == SYSTEM_PROMPT
    assert "Romanized" in SYSTEM_PROMPT and "DZONGKHA" in SYSTEM_PROMPT


def test_history_is_trimmed_and_starts_with_user():
    a = Fake("a")
    e = ChatEngine(settings(MAX_HISTORY_TURNS=2, MAX_MESSAGE_CHARS=10), [a])
    msgs = []
    for i in range(10):
        msgs += [user(f"question number {i}"), {"role": "assistant", "content": "x" * 50}]
    msgs.append(user("last question here"))
    e.reply(msgs)
    sent = a.last_messages
    assert len(sent) <= 5 and sent[0]["role"] == "user"
    assert sent[-1]["content"] == "last question here"  # newest message is not cut at 10 chars
    assert all(len(m["content"]) <= 10 for m in sent[:-1])


def test_followup_history_reaches_provider():
    a = Fake("a")
    ChatEngine(settings(), [a]).reply(
        [user("What is ML?"), {"role": "assistant", "content": DZ}, user("Give an example")]
    )
    assert [m["role"] for m in a.last_messages] == ["user", "assistant", "user"]


# ---- translation ---------------------------------------------------------- #
def test_script_ratio():
    assert dzongkha_script_ratio(DZ) > 0.95
    assert dzongkha_script_ratio("Hello world") == 0.0
    assert dzongkha_script_ratio("12345") == 1.0
    assert dzongkha_script_ratio(DZ + " (machine learning)") > 0.6


class FakeTranslator(GoogleTranslator):
    def __init__(self, s):
        super().__init__(s, usage_file=Path("/nonexistent/x.json"))
        self.sent = []

    def to_dzongkha(self, text):
        self.sent.append(text)
        return DZ


def test_translate_only_when_answer_is_not_already_dzongkha():
    s = settings(ENABLE_GOOGLE_TRANSLATE="true", GOOGLE_TRANSLATE_API_KEY="k")
    t = FakeTranslator(s)
    # good Dzongkha answer: left alone
    r = ChatEngine(s, [Fake("a", DZ)], t).reply([user("hi")])
    assert not r.translated and t.sent == []
    # English answer to a non-English request: translated once
    r = ChatEngine(s, [Fake("a", "This is an English answer.")], t).reply([user("hi")])
    assert r.translated and r.text == DZ and len(t.sent) == 1
    # explicit English request: never translated
    r = ChatEngine(s, [Fake("a", "English answer.")], t).reply([user("answer in English: hi")])
    assert not r.translated and r.text == "English answer."


def test_translate_disabled_by_default():
    assert not GoogleTranslator(settings()).should_translate("English text", False)


def test_translate_budget_blocks_requests(tmp_path):
    s = settings(ENABLE_GOOGLE_TRANSLATE="true", GOOGLE_TRANSLATE_API_KEY="k", TRANSLATE_MONTHLY_CHAR_BUDGET=5)
    tr = GoogleTranslator(s, usage_file=tmp_path / "u.json")
    assert tr.to_dzongkha("this is longer than five characters") is None  # no HTTP call made


# ---- cost guards ----------------------------------------------------------- #
def test_openrouter_non_free_model_is_refused():
    s = settings(OPENAI_COMPAT_BASE_URL="https://openrouter.ai/api/v1", OPENAI_COMPAT_API_KEY="k",
                 OPENAI_COMPAT_MODEL="openai/gpt-4o")
    assert not OpenAICompatProvider(s).is_configured()
    s2 = settings(OPENAI_COMPAT_BASE_URL="https://openrouter.ai/api/v1", OPENAI_COMPAT_API_KEY="k",
                  OPENAI_COMPAT_MODEL="meta-llama/llama-3.3-70b-instruct:free")
    assert OpenAICompatProvider(s2).is_configured()


def test_gemini_without_key_is_not_configured():
    assert not GeminiProvider(settings()).is_configured()


def test_unknown_provider_names_ignored():
    assert settings(PROVIDER_ORDER="gemini, bogus ,ollama,gemini").provider_order == ("gemini", "ollama")


def test_no_hardcoded_secrets_in_source():
    root = Path(__file__).resolve().parent.parent
    for f in list(root.glob("*.py")) + list(root.glob("dzongkha_bot/*.py")):
        text = f.read_text(encoding="utf-8")
        assert "AIza" not in text, f


def test_search_grounding_setting_and_prompt_rules():
    assert settings().gemini_use_search is True
    assert settings(GEMINI_USE_GOOGLE_SEARCH="false").gemini_use_search is False
    assert "ANY question" in SYSTEM_PROMPT and "ACCURACY" in SYSTEM_PROMPT
