"""Local translation backend using Hugging Face `transformers` (NLLB family).

Runs fully offline after the first download. No API key, no cost, no cloud call
at translation time. Language codes: eng_Latn (English), dzo_Tibt (Dzongkha).

NOTE: this loads whichever model id you configure. The default models are
CST-published (College of Science and Technology, Bhutan) NLLB fine-tunes and
Meta's stock NLLB-200-distilled-600M. NONE of them is verified to be GovTech's
deployed model, and GovTech has not published its weights.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional, Set, Tuple

log = logging.getLogger("dzongkha_bot")

CODES = {"en": "eng_Latn", "dz": "dzo_Tibt"}


class BackendError(Exception):
    """A translation backend could not load or run."""


class HFBackend:
    def __init__(
        self,
        name: str,
        model_id: str,
        pairs: Set[Tuple[str, str]],
        device: str = "cpu",
        local_files_only: bool = False,
        beams: int = 4,
        max_new_tokens: int = 256,
        batch_size: int = 8,
    ):
        self.name = name
        self.model_id = model_id
        self.pairs = set(pairs)
        self.device_pref = device
        self.local_files_only = local_files_only
        self.beams = beams
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self._tok = None
        self._model = None
        self._device = "cpu"
        self._lock = threading.Lock()

    def supports(self, src: str, tgt: str) -> bool:
        return (src, tgt) in self.pairs

    def describe(self) -> str:
        return f"{self.name} ({self.model_id})"

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as e:
                raise BackendError(
                    "torch/transformers/sentencepiece are not installed "
                    "(run: pip install -r requirements.txt)"
                ) from e
            try:
                log.info("Loading translation model %s ...", self.model_id)
                tok = AutoTokenizer.from_pretrained(self.model_id, local_files_only=self.local_files_only)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_id, low_cpu_mem_usage=True, local_files_only=self.local_files_only
                )
                want = self.device_pref
                if want == "auto":
                    want = "cuda" if torch.cuda.is_available() else "cpu"
                if want == "cuda" and not torch.cuda.is_available():
                    log.warning("CUDA requested but not available; using CPU")
                    want = "cpu"
                model.eval()
                model.to(want)
                if want == "cuda":
                    model.half()  # fp16 keeps the 600M model near 1.2 GB of VRAM
                for code in CODES.values():
                    if tok.convert_tokens_to_ids(code) in (None, tok.unk_token_id):
                        raise BackendError(f"tokenizer of {self.model_id} has no {code} token")
                self._tok, self._model, self._device = tok, model, want
                log.info("Loaded %s on %s", self.model_id, want)
            except BackendError:
                raise
            except Exception as e:  # download failure, offline, OOM, bad repo...
                raise BackendError(f"{type(e).__name__}: {str(e)[:200]}") from e

    def translate_batch(self, texts: List[str], src: str, tgt: str) -> List[str]:
        if not self.supports(src, tgt):
            raise BackendError(f"{self.name} does not support {src}->{tgt}")
        self.load()
        import torch

        tok, model = self._tok, self._model
        tok.src_lang = CODES[src]
        bos = tok.convert_tokens_to_ids(CODES[tgt])
        out: List[str] = []
        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
                enc = {k: v.to(self._device) for k, v in enc.items()}
                with torch.inference_mode():
                    gen = model.generate(
                        **enc,
                        forced_bos_token_id=bos,
                        max_new_tokens=self.max_new_tokens,
                        num_beams=self.beams,
                        no_repeat_ngram_size=4,
                    )
                out.extend(s.strip() for s in tok.batch_decode(gen, skip_special_tokens=True))
        except Exception as e:
            raise BackendError(f"{type(e).__name__}: {str(e)[:200]}") from e
        return out
