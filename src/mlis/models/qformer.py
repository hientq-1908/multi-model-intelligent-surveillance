"""Q-Former: learnable queries that compress image tokens and give retrieval embeddings.

This is the interface and a minimal working implementation with the right shapes.
It has no losses.
ITC, ITM and ITG (stage 1) are built on top of it in ``mlis.train.stage1``.
Layer and parameter names follow BERT, so ``load_bert_weights`` can initialise it from
BERT-base the way BLIP-2 does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

BERT_BASE = "bert-base-uncased"


class QFormerConfig(BaseModel):
    """Defaults describe the planned model: 64 queries of width 768 on BERT-base."""

    model_config = ConfigDict(extra="forbid")

    num_queries: int = Field(default=64, ge=1)
    hidden_size: int = Field(default=768, ge=1)
    num_layers: int = Field(default=12, ge=1)
    num_heads: int = Field(default=12, ge=1)
    intermediate_size: int = Field(default=3072, ge=1)
    vision_dim: int = Field(default=1024, ge=1, description="Channel size of the image tokens.")
    embed_dim: int = Field(default=256, ge=1, description="Retrieval embedding size.")
    cross_attention_every: int = Field(default=2, ge=1)
    vocab_size: int = Field(default=30522, ge=1)
    max_text_length: int = Field(default=512, ge=1)
    layer_norm_eps: float = 1e-12


@dataclass(frozen=True)
class QFormerOutput:
    """``query_tokens`` is ``[B, num_queries, hidden_size]``.

    ``image_embed`` and ``text_embed`` are L2-normalised ``[B, embed_dim]`` retrieval vectors.
    ``text_embed`` is None when no text was given.
    """

    query_tokens: Tensor
    image_embed: Tensor
    text_embed: Tensor | None = None


class QFormerBase(nn.Module, ABC):
    """Interface the stage code and the LLM stages depend on."""

    @property
    @abstractmethod
    def num_queries(self) -> int: ...

    @property
    @abstractmethod
    def hidden_size(self) -> int: ...

    @abstractmethod
    def encode_image(self, image_tokens: Tensor) -> QFormerOutput:
        """``image_tokens`` is ``[B, N, vision_dim]``; text side stays empty."""

    @abstractmethod
    def encode_text(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Return the L2-normalised text retrieval embedding ``[B, embed_dim]``."""

    def forward(
        self,
        image_tokens: Tensor,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> QFormerOutput:
        output = self.encode_image(image_tokens)
        if input_ids is None:
            return output
        text_embed = self.encode_text(input_ids, attention_mask)
        return QFormerOutput(output.query_tokens, output.image_embed, text_embed)


class _Attention(nn.Module):
    """Multi-head attention with BERT parameter names, usable for self and cross attention."""

    def __init__(self, hidden: int, heads: int, kv_dim: int, eps: float) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError(f"hidden_size {hidden} must be divisible by num_heads {heads}")
        self.heads = heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(kv_dim, hidden)
        self.value = nn.Linear(kv_dim, hidden)
        self.dense = nn.Linear(hidden, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)

    def forward(
        self, x: Tensor, context: Tensor | None = None, mask: Tensor | None = None
    ) -> Tensor:
        context = x if context is None else context
        batch, length, hidden = x.shape
        head_dim = hidden // self.heads

        def split(t: Tensor) -> Tensor:
            return t.view(batch, -1, self.heads, head_dim).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            split(self.query(x)),
            split(self.key(context)),
            split(self.value(context)),
            attn_mask=mask,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, hidden)
        return self.LayerNorm(self.dense(attended) + x)


class _FeedForward(nn.Module):
    def __init__(self, hidden: int, intermediate: int, eps: float) -> None:
        super().__init__()
        self.intermediate = nn.Linear(hidden, intermediate)
        self.output = nn.Linear(intermediate, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.LayerNorm(self.output(F.gelu(self.intermediate(x))) + x)


class _Layer(nn.Module):
    """Self-attention shared by queries and text, cross-attention for queries on some layers.

    Queries and text use separate feed-forward blocks, as in BLIP-2.
    """

    def __init__(self, config: QFormerConfig, *, has_cross_attention: bool) -> None:
        super().__init__()
        hidden, heads, eps = config.hidden_size, config.num_heads, config.layer_norm_eps
        self.attention = _Attention(hidden, heads, hidden, eps)
        self.crossattention = (
            _Attention(hidden, heads, config.vision_dim, eps) if has_cross_attention else None
        )
        self.ffn_text = _FeedForward(hidden, config.intermediate_size, eps)
        self.ffn_query = _FeedForward(hidden, config.intermediate_size, eps)

    def forward_queries(self, queries: Tensor, image_tokens: Tensor) -> Tensor:
        queries = self.attention(queries)
        if self.crossattention is not None:
            queries = self.crossattention(queries, context=image_tokens)
        return self.ffn_query(queries)

    def forward_text(self, text: Tensor, mask: Tensor | None) -> Tensor:
        return self.ffn_text(self.attention(text, mask=mask))


class QFormer(QFormerBase):
    """Minimal Q-Former: image path from queries, text path from a BERT-style encoder.

    The two paths share self-attention weights and never see each other, which is the
    uni-modal setup that ITC needs.
    Multi-modal attention for ITM and ITG is added by stage 1.
    """

    def __init__(self, config: QFormerConfig | None = None) -> None:
        super().__init__()
        self.config = config or QFormerConfig()
        cfg = self.config
        self.query_tokens = nn.Parameter(torch.empty(1, cfg.num_queries, cfg.hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embeddings = nn.Embedding(cfg.max_text_length, cfg.hidden_size)
        self.embeddings_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layers = nn.ModuleList(
            _Layer(cfg, has_cross_attention=i % cfg.cross_attention_every == 0)
            for i in range(cfg.num_layers)
        )
        self.image_proj = nn.Linear(cfg.hidden_size, cfg.embed_dim)
        self.text_proj = nn.Linear(cfg.hidden_size, cfg.embed_dim)
        self.apply(self._init_weights)

    @property
    def num_queries(self) -> int:
        return self.config.num_queries

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def encode_image(self, image_tokens: Tensor) -> QFormerOutput:
        queries = self.query_tokens.expand(image_tokens.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer.forward_queries(queries, image_tokens)
        pooled = F.normalize(self.image_proj(queries).mean(dim=1), dim=-1)
        return QFormerOutput(query_tokens=queries, image_embed=pooled)

    def encode_text(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        length = input_ids.shape[1]
        if length > self.config.max_text_length:
            raise ValueError(f"text length {length} exceeds {self.config.max_text_length}")
        positions = torch.arange(length, device=input_ids.device)
        text = self.embeddings_norm(
            self.word_embeddings(input_ids) + self.position_embeddings(positions)
        )
        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].bool()
        for layer in self.layers:
            text = layer.forward_text(text, mask)
        return F.normalize(self.text_proj(text[:, 0]), dim=-1)

    @torch.no_grad()
    def load_bert_weights(self, bert_state: Mapping[str, Tensor]) -> list[str]:
        """Copy BERT-base weights into the text embeddings and every layer.

        ``bert_state`` is the ``state_dict`` of a ``transformers.BertModel``.
        Self-attention and the text feed-forward copy over directly; the query feed-forward
        starts as a copy of the text one, as in BLIP-2.
        Cross-attention, queries and projections keep their random initialisation.
        Returns the names of the Q-Former parameters that were left untouched.
        """
        target = self.state_dict()
        copied: set[str] = set()

        def copy(dst: str, src: str) -> None:
            tensor = bert_state[src]
            if target[dst].shape != tensor.shape:
                raise ValueError(f"{dst} {tuple(target[dst].shape)} vs {src} {tuple(tensor.shape)}")
            target[dst].copy_(tensor)
            copied.add(dst)

        copy("word_embeddings.weight", "embeddings.word_embeddings.weight")
        copy("position_embeddings.weight", "embeddings.position_embeddings.weight")
        copy("embeddings_norm.weight", "embeddings.LayerNorm.weight")
        copy("embeddings_norm.bias", "embeddings.LayerNorm.bias")
        for i in range(self.config.num_layers):
            base = f"encoder.layer.{i}"
            for part in ("query", "key", "value"):
                for kind in ("weight", "bias"):
                    copy(
                        f"layers.{i}.attention.{part}.{kind}",
                        f"{base}.attention.self.{part}.{kind}",
                    )
            for kind in ("weight", "bias"):
                copy(f"layers.{i}.attention.dense.{kind}", f"{base}.attention.output.dense.{kind}")
                copy(
                    f"layers.{i}.attention.LayerNorm.{kind}",
                    f"{base}.attention.output.LayerNorm.{kind}",
                )
                for ffn in ("ffn_text", "ffn_query"):
                    copy(
                        f"layers.{i}.{ffn}.intermediate.{kind}", f"{base}.intermediate.dense.{kind}"
                    )
                    copy(f"layers.{i}.{ffn}.output.{kind}", f"{base}.output.dense.{kind}")
                    copy(f"layers.{i}.{ffn}.LayerNorm.{kind}", f"{base}.output.LayerNorm.{kind}")
        return sorted(set(target) - copied)

    @classmethod
    def from_bert_base(cls, config: QFormerConfig | None = None, name: str = BERT_BASE) -> QFormer:
        """Build a Q-Former and initialise it from a pretrained BERT-base checkpoint."""
        from transformers import BertModel

        model = cls(config)
        bert = BertModel.from_pretrained(name, add_pooling_layer=False)
        model.load_bert_weights(bert.state_dict())
        return model


__all__ = ["BERT_BASE", "QFormer", "QFormerBase", "QFormerConfig", "QFormerOutput"]
