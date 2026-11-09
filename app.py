"""Dzongkha AI Chatbot - simple, professional Streamlit UI.

Welcome screen: only the greeting (Kuzuzangpo La / ཀུ་ཟུགས་བཟང་པོ) and a short
Dzongkha line. After the first message the conversation takes over. The chat
logic (engine, Ollama, translation pipeline) is untouched; the active provider
is set in .env and logged to the console.
"""
import logging

import streamlit as st

from dzongkha_bot import engine as eng
from dzongkha_bot import strings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

st.set_page_config(page_title="Dzongkha AI Chatbot", page_icon="🏔️", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Tibetan:wght@400;600&family=Inter:wght@400;500;600&display=swap');

:root {
  --red: #7A1F1F;
  --gold: #C9A24B;
  --cream: #FBF7EF;
  --ink: #231F1C;
  --muted: #6F665E;
  --line: #E9E0D1;
}
html, body, .stApp, [data-testid="stAppViewContainer"] { background: var(--cream) !important; }
body, .stApp, button { font-family: "Inter", "Noto Sans Tibetan", system-ui, sans-serif; color: var(--ink); }

/* Dzongkha-capable fonts for all text */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown div, textarea, [data-testid="stChatInputTextArea"] {
  font-family: "Noto Sans Tibetan", "Jomolhari", "Microsoft Himalaya", "Noto Serif Tibetan", "Tibetan Machine Uni", "Inter", system-ui, sans-serif !important;
}

header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
.block-container { max-width: 800px; padding: 0 1.25rem 7rem; }

/* ---- Header ---- */
.topbar { display:flex; align-items:center; gap:10px; padding:16px 0 12px; margin-bottom:8px;
  border-bottom:2px solid var(--gold); }
.topbar .mark { width:10px; height:26px; background:var(--red); border-radius:2px; }
.topbar .name { font-family:"Inter",sans-serif; font-size:20px; font-weight:600; color:var(--red); letter-spacing:.01em; }

/* ---- Welcome ---- */
.welcome-en { text-align:center; font-family:"Inter",sans-serif; font-weight:700; font-size:48px; line-height:1.25;
  color:var(--red); margin:15vh 0 .2rem; letter-spacing:.01em; }
.welcome-dz { text-align:center; font-family:"Noto Sans Tibetan","Jomolhari","Microsoft Himalaya","Noto Serif Tibetan",serif; font-size:44px; font-weight:700;
  line-height:1.9; color:var(--ink); margin:0 0 .6rem; }
.welcome-rule { width:56px; height:2px; background:var(--gold); margin:0 auto 1.1rem; }
.st-key-greeting, .st-key-greeting * { text-align:center; }
.st-key-greeting p { font-size:26px !important; font-weight:700 !important; line-height:2; color:var(--ink); }

/* ---- Chat bubbles ---- */
[data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:14px; padding:.85rem 1.1rem;
  margin-bottom:.75rem; background:#fff; overflow-wrap:anywhere; word-break:break-word; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:#F6EEDC; border-color:#E8D8AE; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { font-size:22px !important; font-weight:600 !important; line-height:2; margin-bottom:.3rem; }
[data-testid="stChatMessageAvatarUser"] { background:var(--gold) !important; }
[data-testid="stChatMessageAvatarAssistant"] { background:var(--red) !important; }

/* ---- Input + send button ---- */
[data-testid="stChatInput"], [data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea { background:#fff !important; }
[data-testid="stChatInput"] { border:1.5px solid var(--line); border-radius:12px; }
[data-testid="stChatInput"]:focus-within, [data-testid="stChatInput"] > div:focus-within { border-color:var(--gold) !important; box-shadow:0 0 0 2px rgba(201,162,75,.3) !important; outline:none !important; }
[data-testid="stChatInput"] textarea:focus { outline:none !important; box-shadow:none !important; }
[data-testid="stChatInput"] textarea { font-size:20px !important; font-weight:600; line-height:1.8; }
[data-testid="stChatInput"] button { background:var(--red) !important; color:#fff !important; border-radius:9px; }
[data-testid="stChatInput"] button:hover { background:#5E1616 !important; }
[data-testid="stChatInput"] button svg { fill:#fff; }
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background:var(--cream) !important; }

/* ---- Buttons / labels ---- */
div[data-testid="stButton"] > button { font-size:17px; font-weight:600; background:transparent; color:var(--red); border:1px solid var(--line);
  border-radius:8px; padding:.3rem .8rem; }
div[data-testid="stButton"] > button:hover { border-color:var(--gold); color:var(--red); }

@media (max-width:640px) {
  .welcome-en { font-size:38px; margin-top:10vh; }
  .welcome-dz { font-size:36px; }
  .st-key-greeting p { font-size:22px !important; }
  [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { font-size:20px !important; }
  .block-container { padding: 0 .8rem 7rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def get_engine():
    # Shared across sessions on purpose: it holds the loaded translation models
    # and any API rate-limit cool-downs.
    engine = eng.build_engine()
    warm = getattr(engine, "warm_up", None)
    if warm:
        warm()  # loads the English->Dzongkha model once, so the first answer is not slow
    return engine


if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str}]

# ---- Header --------------------------------------------------------------- #
st.markdown(
    '<div class="topbar"><div class="mark"></div><div class="name">Dzongkha AI Chatbot</div></div>',
    unsafe_allow_html=True,
)

if st.session_state.messages:
    _, right = st.columns([5, 1.6])
    if right.button("New chat", key="new_chat"):
        st.session_state.messages = []
        st.rerun()
else:
    # Welcome screen: heading + short Dzongkha greeting only (display only; never sent to the model).
    st.markdown(
        '<div class="welcome-en">Kuzuzangpo La</div>'
        '<div class="welcome-dz">ཀུ་ཟུགས་བཟང་པོ</div>'
        '<div class="welcome-rule"></div>',
        unsafe_allow_html=True,
    )
    with st.container(key="greeting"):
        st.markdown(strings.WELCOME)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

with st.spinner("…"):  # first start only: loads the translation model
    engine = get_engine()


def model_history(messages):
    """History sent to the model: drops error notices and the question each one
    failed to answer, so errors never pollute later requests."""
    out = []
    for m in messages:
        if m.get("error"):
            if out and out[-1]["role"] == "user":
                out.pop()
            continue
        item = {"role": m["role"], "content": m["content"]}
        if m.get("en") is not None:
            item["en"] = m["en"]  # English version, used by the translation pipeline
        out.append(item)
    return out


prompt = st.chat_input(strings.INPUT_PLACEHOLDER)
if prompt and prompt.strip():
    st.session_state.messages.append({"role": "user", "content": prompt.strip()})
    with st.chat_message("user"):
        st.markdown(prompt.strip())
    with st.chat_message("assistant"):
        with st.spinner("…"):
            reply = engine.reply(model_history(st.session_state.messages))
        st.markdown(reply.text)
    if reply.user_en is not None:
        st.session_state.messages[-1]["en"] = reply.user_en
    st.session_state.messages.append(
        {"role": "assistant", "content": reply.text, "en": reply.en, "error": reply.error is not None}
    )
    st.rerun()  # redraw from saved state so the welcome screen gives way to the conversation
