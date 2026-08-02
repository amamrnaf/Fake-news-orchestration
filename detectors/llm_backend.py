"""Unified LLM calling interface used by T2, T3, the router, and the all-tool
advisor baseline. Two interchangeable backends:

  - LocalHFBackend: open instruction-tuned 7-8B model in 4-bit quantisation
    (Sec 4.4), run on-device (e.g. RTX 4070 laptop GPU).
  - APIBackend: any OpenAI-compatible chat-completions endpoint (hosted API,
    or a local server such as vLLM/text-generation-inference exposing the
    same protocol), useful for pilot testing before committing to a local
    checkpoint.

Both backends are deterministic (temperature=0) and report latency, generated
token count, and peak GPU memory so the efficiency tables (Sec 4.5, Table 11)
can be filled from real measurements rather than estimates.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    generated_tokens: int
    latency_s: float
    peak_vram_mb: float


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str | None = None, max_new_tokens: int = 256) -> LLMResponse: ...


class LocalHFBackend(LLMBackend):
    """4-bit quantised local instruction model (Sec 4.4)."""

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        load_in_4bit: bool = True,
        max_input_tokens: int = 4096,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, **quant_kwargs
        )
        self.model.eval()

    def generate(self, prompt: str, system: str | None = None, max_new_tokens: int = 256) -> LLMResponse:
        torch = self.torch
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        input_ids = input_ids[:, -self.max_input_tokens :]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - start
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        )

        new_tokens = output_ids[0][input_ids.shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return LLMResponse(
            text=text, generated_tokens=int(new_tokens.shape[0]), latency_s=latency, peak_vram_mb=peak_vram_mb
        )


class APIBackend(LLMBackend):
    """OpenAI-compatible chat-completions backend."""

    def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, prompt: str, system: str | None = None, max_new_tokens: int = 256) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=max_new_tokens,
        )
        latency = time.perf_counter() - start
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        generated_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        return LLMResponse(text=text, generated_tokens=generated_tokens, latency_s=latency, peak_vram_mb=0.0)


def build_llm_backend(cfg: dict) -> LLMBackend:
    """cfg example (see configs/default.yaml -> llm):
    {"backend": "local", "model_name": "...", "load_in_4bit": true}
    {"backend": "api", "model_name": "...", "base_url": "...", "api_key": "..."}
    """
    backend = cfg["backend"]
    if backend == "local":
        return LocalHFBackend(
            model_name=cfg["model_name"],
            load_in_4bit=cfg.get("load_in_4bit", True),
            max_input_tokens=cfg.get("max_input_tokens", 4096),
        )
    if backend == "api":
        return APIBackend(model_name=cfg["model_name"], base_url=cfg.get("base_url"), api_key=cfg.get("api_key"))
    raise ValueError(f"Unknown llm backend {backend!r}")
