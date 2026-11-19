"""Qwen3 implementation of the LLM wrapper, loaded through Hugging Face transformers."""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, PreTrainedModel

from mlis.models.llm import QWEN3_MAIN, LLMOutput, LLMWrapper


class Qwen3Wrapper(LLMWrapper):
    """Qwen3 decoder with frozen base weights by default.

    ``Qwen3-4B-Instruct-2507`` has no thinking mode, so no chat-template switch is needed.
    The smoke model ``Qwen3-0.6B`` does have one; the stage code must render its prompts
    with thinking off.
    Adding LoRA is the job of stage 2/3 training code, not of this wrapper.
    """

    def __init__(self, model: PreTrainedModel, *, freeze: bool = True) -> None:
        super().__init__()
        self.model = model
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str = QWEN3_MAIN,
        *,
        dtype: torch.dtype = torch.bfloat16,
        freeze: bool = True,
    ) -> Qwen3Wrapper:
        return cls(
            AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=dtype), freeze=freeze
        )

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> LLMOutput:
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return LLMOutput(logits=out.logits, loss=out.loss if labels is not None else None)

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        max_new_tokens: int = 64,
    ) -> Tensor:
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
