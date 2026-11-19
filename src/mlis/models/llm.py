"""LLM wrapper interface: the only way stage code touches the language model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn

QWEN3_MAIN = "Qwen/Qwen3-4B-Instruct-2507"
QWEN3_SMOKE = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True)
class LLMOutput:
    """``logits`` is ``[B, T, vocab]``; ``loss`` is set only when labels were given."""

    logits: Tensor
    loss: Tensor | None = None


class LLMWrapper(nn.Module, ABC):
    """Decoder-only LLM that accepts input embeddings, so visual tokens can be spliced in.

    ``labels`` use -100 at positions that must not count towards the loss.
    """

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Width of the token embeddings, which the projector must output."""

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        """Return token embeddings ``[B, T, hidden_size]``."""

    @abstractmethod
    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> LLMOutput: ...

    @abstractmethod
    def generate(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        max_new_tokens: int = 64,
    ) -> Tensor:
        """Greedy decoding from embeddings; returns only the new token ids ``[B, new]``."""
