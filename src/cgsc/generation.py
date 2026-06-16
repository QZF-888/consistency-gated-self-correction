from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float = 0.7
    top_p: float = 0.95
    load_in_4bit: bool = True


class HFGenerator:
    def __init__(self, model_id: str, config: GenerationConfig, token: str | None = None, trust_remote_code: bool = True):
        self.config = config
        tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
        model_kwargs = {"device_map": "auto", "torch_dtype": torch.float16, "trust_remote_code": trust_remote_code}
        if config.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        if token:
            tokenizer_kwargs["token"] = token
            model_kwargs["token"] = token
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.eval()
        self.eos_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []

    def _chat_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    @torch.no_grad()
    def generate(self, prompt: str, do_sample: bool = False, n_return: int = 1) -> str | list[str]:
        inputs = self.tokenizer(self._chat_text(prompt), return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        kwargs = {"max_new_tokens": self.config.max_new_tokens, "do_sample": do_sample, "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": self.eos_ids or self.tokenizer.eos_token_id}
        if do_sample:
            kwargs.update({"temperature": self.config.temperature, "top_p": self.config.top_p, "num_return_sequences": n_return})
        output_ids = self.model.generate(**inputs, **kwargs)
        input_len = inputs["input_ids"].shape[-1]
        texts = [self.tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True) for i in range(output_ids.shape[0])]
        return texts[0] if n_return == 1 else texts
