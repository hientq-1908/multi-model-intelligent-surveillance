"""Projector from Q-Former query tokens into the LLM input embedding space."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Projector(nn.Module, ABC):
    """Contract: ``forward([B, Q, in_dim])`` returns ``[B, Q, llm_hidden_size]``."""

    in_dim: int
    llm_hidden_size: int

    @abstractmethod
    def forward(self, query_tokens: Tensor) -> Tensor: ...


class MLPProjector(Projector):
    """Two linear layers with a GELU in between, the LLaVA-1.5 design."""

    def __init__(self, in_dim: int, llm_hidden_size: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.llm_hidden_size = llm_hidden_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, query_tokens: Tensor) -> Tensor:
        return self.net(query_tokens)
