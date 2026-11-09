"""Headless UI test with Streamlit's AppTest and a fake engine (no network)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from dzongkha_bot import engine as eng
from dzongkha_bot import strings
from dzongkha_bot.engine import Reply

ANSWER = "ཨིན། འདི་ཡིག་ཆ་ཨིན།"


class FakeEngine:
    def __init__(self):
        self.seen = []

    def reply(self, messages):
        self.seen.append([dict(m) for m in messages])
        return Reply(ANSWER, provider="fake")


def run_app(monkeypatch, engine):
    import streamlit as st

    st.cache_resource.clear()  # get_engine() is cached; each test needs its own fake
    monkeypatch.setattr(eng, "build_engine", lambda *a, **k: engine)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    return at


def texts(at):
    return [m.value for m in at.markdown]


def test_initial_screen_is_minimal(monkeypatch):
    at = run_app(monkeypatch, FakeEngine())
    assert not at.exception
    assert strings.WELCOME in texts(at)
    assert len(at.chat_input) == 1
    assert len(at.button) == 0 and len(at.sidebar.children) == 0
    assert len(at.selectbox) == 0 and len(at.text_input) == 0


def test_conversation_and_history(monkeypatch):
    fake = FakeEngine()
    at = run_app(monkeypatch, fake)
    at.chat_input[0].set_value("What is ML?").run()
    at.chat_input[0].set_value("Give an example").run()
    assert not at.exception
    assert ANSWER in texts(at)
    # second call carried the whole session history, ending with the new question
    roles = [m["role"] for m in fake.seen[1]]
    assert roles == ["user", "assistant", "user"]
    assert fake.seen[1][-1]["content"] == "Give an example"


def test_error_notice_not_sent_back_as_history(monkeypatch):
    class Failing(FakeEngine):
        def __init__(self):
            super().__init__()
            self.n = 0

        def reply(self, messages):
            self.seen.append([dict(m) for m in messages])
            self.n += 1
            if self.n == 1:
                return Reply(strings.ERR_RATE_LIMIT, error="rate_limit")
            return Reply(ANSWER, provider="fake")

    fake = Failing()
    at = run_app(monkeypatch, fake)
    at.chat_input[0].set_value("first").run()
    at.chat_input[0].set_value("second").run()
    assert [m["content"] for m in fake.seen[1]] == ["second"]


def test_english_versions_are_stored_and_sent_back(monkeypatch):
    class Pipe(FakeEngine):
        def reply(self, messages):
            self.seen.append([dict(m) for m in messages])
            return Reply(ANSWER, provider="fake", en="English answer", user_en="English question")

    fake = Pipe()
    at = run_app(monkeypatch, fake)
    at.chat_input[0].set_value("hello").run()
    at.chat_input[0].set_value("again").run()
    first_two = fake.seen[1][:2]
    assert first_two[0]["en"] == "English question" and first_two[1]["en"] == "English answer"
