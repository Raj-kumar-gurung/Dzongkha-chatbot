"""Quick configuration check.  python check_setup.py

Sends NO chat message. It checks installed packages, GPU visibility, whether the
translation models are already in the local Hugging Face cache, and whether
Ollama is running with your model.
"""
import requests

from dzongkha_bot.config import load_settings

s = load_settings()
print("Mode:", s.pipeline_mode, "| chat providers:", ", ".join(s.provider_order) or "(none)")

if s.pipeline_mode == "nllb":
    try:
        import torch, transformers, sentencepiece  # noqa: F401, E401

        print(f"[translate] torch {torch.__version__}, transformers {transformers.__version__}")
        print("[translate] CUDA GPU visible:", torch.cuda.is_available(), "| using device:", s.translate_device)
    except ImportError as e:
        print("[translate] MISSING PACKAGE:", e.name, "-> run: pip install -r requirements.txt")
    try:
        from huggingface_hub import try_to_load_from_cache

        ids = []
        if "cst" in s.translator_order:
            ids += [s.cst_en_to_dz_model, s.cst_dz_to_en_model]
        if "nllb" in s.translator_order:
            ids.append(s.nllb_stock_model)
        for mid in ids:
            hit = try_to_load_from_cache(mid, "config.json")
            print(f"[translate] {mid}: {'downloaded' if isinstance(hit, str) else 'NOT downloaded (run: python download_models.py)'}")
    except Exception as e:
        print("[translate] cache check failed:", type(e).__name__)

if "ollama" in s.provider_order:
    try:
        have = [m["name"] for m in requests.get(f"{s.ollama_host}/api/tags", timeout=5).json().get("models", [])]
        print(f"[ollama] running; models: {have or 'none'}")
        print("[ollama] configured model present:", any(h.startswith(s.ollama_model) for h in have), f"({s.ollama_model})")
    except Exception:
        print("[ollama] not reachable at", s.ollama_host, "- start Ollama")

if "gemini" in s.provider_order:
    print("[gemini] key set:", bool(s.gemini_api_key), "| model:", s.gemini_model)
if "openai_compat" in s.provider_order:
    print("[openai_compat] configured:", bool(s.compat_base_url and s.compat_api_key and s.compat_model))
