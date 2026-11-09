"""Download and test-load the translation models once, so the app starts fast and
can later run offline.   python download_models.py

Hugging Face caches them (Windows: %USERPROFILE%\\.cache\\huggingface). Each model
is roughly 2.5 GB if it matches the base NLLB-200-distilled-600M checkpoint
(not verified for the CST fine-tunes). Models are loaded one at a time.
"""
import gc
import logging

from dzongkha_bot.config import load_settings
from dzongkha_bot.hf_translator import BackendError
from dzongkha_bot.translation_service import build_translation_service

logging.basicConfig(level=logging.INFO, format="%(message)s")
s = load_settings()
service = build_translation_service(s)
if not service.backends:
    raise SystemExit("TRANSLATOR_ORDER is empty. Set it to cst,nllb in .env")

for b in service.backends:
    print(f"\n== {b.describe()} ==")
    try:
        b.load()
        print("OK, loaded.")
    except BackendError as e:
        print("FAILED:", e)
    b._model = b._tok = None  # free memory before the next one
    gc.collect()
print("\nDone. Set TRANSLATE_LOCAL_FILES_ONLY=true in .env if you want strictly offline use.")
