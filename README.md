# Dzongkha Chatbot (Streamlit + local Ollama + local translation)

## 2. Ask in English, Dzongkha, or a mix. Answers are shown in Dzongkha script. The screen has only a Dzongkha welcome message, the conversation, and a message box with a send button.


## 2. What is used instead (free, local)

| Component | Model | Status |
|---|---|---|
| Chat (English) | Ollama `llama3.2:3b` (yours) | Local, free. Small model: expect simple answers and some errors. |
| Translation, preferred | CST fine-tunes on Hugging Face: [`KarmaCST/nllb-200-distilled-600M-dz-to-en`](https://huggingface.co/KarmaCST/nllb-200-distilled-600M-dz-to-en) and `KarmaCST/nllb-200-distilled-600M-en-to-dz` | Public, licence `cc-by-nc-4.0`. I read the dz-to-en model card (fine-tune of NLLB-200-distilled-600M, no training set named, BLEU 42.9 on its own validation set). The en-to-dz repo appeared in search results but **I could not open its card**. |
| Translation, backup | Meta [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M), codes `eng_Latn` / `dzo_Tibt` | Licence `cc-by-nc-4.0`. Meta calls it a research model, not for production, trained on inputs up to 512 tokens. |

**None of these is GovTech's own fine-tuned model.** The CST models come from the same Bhutanese institution GovTech credits for the initial fine-tuning, but GovTech says 700,000 sentences and CST's page says 30K, so they are not shown to be the same model. Do not describe the output as "GovTech quality".

**Licence:** CC-BY-NC means non-commercial use only. Fine for personal learning and research. If you want to use it inside a commercial product, check the licence and ask the model owners first.

## 3. How it works

1. Detect the language of your message (no model call).
2. Dzongkha (or the Dzongkha part of a mixed message) is translated to English. Pure English is passed through untouched.
3. `llama3.2:3b` answers in English, using the English version of the chat history.
4. The English answer is translated to Dzongkha. Not done if you explicitly ask for English ("answer in English"). A Dzongkha message is never translated twice.

Details: code blocks are never translated, list markers and line breaks are kept, long text is split into sentences (max 40 per message, `TRANSLATE_MAX_SENTENCES`). If a translator returns the wrong script, the next backend is tried. If all fail: a Dzongkha error is shown for your input, or your answer is shown in English with a Dzongkha notice (never a fake translation). Nothing here can use a paid service. Names and numbers usually survive, but a translator can still alter them, so check important details.

## 4. Setup on Windows 11 (PowerShell, VS Code)

```powershell
# 0. Back up your old project first (this project replaces app.py)
Copy-Item -Recurse . ..\dzongkha-chatbot-backup

# 1. Virtual environment
python -m venv venv                               
venv\Scripts\activate                                                                        
# If blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 2. Dependencies (torch is large, about 200 MB for the CPU build)
pip install -r requirements.txt

# 3. Config
Copy-Item .env.example .env

# 4. Ollama (you already have the model; make sure it is there)
ollama pull llama3.2:3b

# 5. Download the translation models once (several GB, needs internet)
python download_models.py

# 6. Check everything
python check_setup.py

# 7. Run
streamlit run app.py
```

In VS Code: open the folder, press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, pick `.venv`, and use the built-in terminal (`Ctrl+``) for the commands above.

**Hardware (16 GB RAM, RTX 3050 4 GB):** each translation model needs roughly 2.5 GB of disk and RAM if it matches the base checkpoint (not verified for the CST models). The English-to-Dzongkha model loads at start, the Dzongkha-to-English one on your first Dzongkha message, and the backup only if needed. By default translation runs on the **CPU** (`TRANSLATE_DEVICE=cpu`) so the GPU stays free for Ollama. You can try `cuda`, but the Windows `pip install torch` wheel is usually CPU-only (see https://pytorch.org/get-started/locally for the GPU build) and 4 GB of VRAM is shared with Ollama. I could not measure speed here; expect seconds per sentence on CPU.

After the first download, set `TRANSLATE_LOCAL_FILES_ONLY=true` for strictly offline translation.

## 5. Testing

What was verified in my build environment: **42 automated offline tests pass** (routing, fallback, script validation, failure handling, long text, code blocks, UI, history, no hardcoded keys). What was **not** verified: any real translation or real Ollama/NLLB output. Hugging Face and PyTorch downloads were blocked there, so Dzongkha quality is unmeasured.

```powershell
pip install pytest
python -m pytest -q tests                  # logic tests, no models needed
python -m eval.run_translation_eval        # translation quality ONLY (needs models)
python -m eval.run_eval                    # chat quality, 26 messages (needs models + Ollama)
```

Translation and chat quality are evaluated separately. `run_translation_eval` writes `eval/translation_results.md` (English to Dzongkha to English round trips, plus Dzongkha to English). Put fluent-speaker reference translations in `eval/translation_cases.py` (`REFERENCE_DZ`) to get a chrF score. `run_eval` writes `eval/results.md`. Both have a review column for a fluent Dzongkha speaker; the automatic checks cannot judge correctness. The Dzongkha test sentences were written by an AI, so replace them.

How to reproduce each scenario:

| Scenario | How |
|---|---|
| English greetings / ordinary chat | `run_eval` cases G1-G4, S1-S4 |
| English question, Dzongkha answer | cases M1-M4, F1-F2 (follow-ups) |
| Dzongkha in, Dzongkha out | cases D1-D4 |
| Mixed English + Dzongkha | cases X1-X3 |
| Long messages | cases L1, L2 (limit: 40 sentences) |
| Translation failure | set `CST_EN_TO_DZ_MODEL=nope/nope` and `TRANSLATOR_ORDER=cst`, restart: you should see the Dzongkha notice plus the English answer |
| Network unavailable | after step 5, set `TRANSLATE_LOCAL_FILES_ONLY=true`, disconnect, run |
| Local-only, no paid keys | default `.env`: `PROVIDER_ORDER=ollama`, empty key fields |

## 6. Optional extras

- **Free Gemini as the chat model:** put `gemini` first in `PROVIDER_ORDER` and set `GEMINI_API_KEY` ([AI Studio](https://aistudio.google.com/apikey)). Better answers, but needs internet, free-tier limits can change, and free-tier prompts may be used by Google. Never enable billing on that project. Ollama stays as the fallback.
- `PIPELINE_MODE=direct`: the chat model writes Dzongkha itself (old behaviour). With `llama3.2:3b` this is expected to be poor.
- Google Cloud Translation (direct mode only) is off by default because it needs a billing-enabled project.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Dzongkha shows as boxes | Windows has Microsoft Himalaya built in. The page also loads Noto Sans Tibetan from Google Fonts (needs internet). |
| Translation notice every time | Run `python check_setup.py`. Probably the packages or models are missing: `pip install -r requirements.txt`, then `python download_models.py`. |
| `download_models.py` fails | Check your internet or proxy. Hugging Face model pages must be reachable. |
| "tokenizer has no dzo_Tibt token" | That model id is not NLLB-compatible. Change it in `.env`, or use `TRANSLATOR_ORDER=nllb`. |
| Out of memory | Keep `TRANSLATE_DEVICE=cpu`, close other apps. |
| First message very slow | The model is loading. Later messages are faster. |
| "AI service not configured" / unavailable | Start Ollama; `ollama list` must show `llama3.2:3b`. |
| Answers wrong or odd | `llama3.2:3b` is small and has no web search. Try a bigger Ollama model, or free Gemini (section 6). Dzongkha wording needs a fluent reviewer. |
| Activation script blocked | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

All Dzongkha UI text other than your welcome message (`dzongkha_bot/strings.py`) was written by an AI and needs review by a fluent speaker.
