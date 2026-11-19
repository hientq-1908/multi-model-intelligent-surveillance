import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from mlis.models.llm import QWEN3_MAIN, QWEN3_SMOKE
from mlis.models.projector import MLPProjector
from mlis.models.qwen3 import Qwen3Wrapper


def tiny_llm(freeze: bool = True) -> Qwen3Wrapper:
    config = Qwen3Config(
        vocab_size=120,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=12,
        max_position_embeddings=64,
    )
    return Qwen3Wrapper(Qwen3ForCausalLM(config), freeze=freeze)


def test_model_names_follow_the_plan():
    assert QWEN3_MAIN == "Qwen/Qwen3-4B-Instruct-2507"
    assert QWEN3_SMOKE == "Qwen/Qwen3-0.6B"


def test_projector_maps_query_tokens_to_llm_width():
    projector = MLPProjector(in_dim=32, llm_hidden_size=48)
    out = projector(torch.randn(2, 8, 32))
    assert out.shape == (2, 8, 48)
    assert projector.llm_hidden_size == 48


def test_llm_exposes_hidden_size_and_embeddings():
    llm = tiny_llm()
    assert (llm.hidden_size, llm.vocab_size) == (48, 120)
    assert llm.embed_tokens(torch.randint(0, 120, (2, 5))).shape == (2, 5, 48)


def test_llm_is_frozen_by_default_and_trainable_on_request():
    assert not any(p.requires_grad for p in tiny_llm().parameters())
    assert all(p.requires_grad for p in tiny_llm(freeze=False).parameters())


def test_forward_returns_logits_and_loss_only_with_labels():
    llm = tiny_llm()
    embeds = torch.randn(2, 6, 48)
    assert llm(embeds).loss is None
    out = llm(embeds, labels=torch.randint(0, 120, (2, 6)))
    assert out.logits.shape == (2, 6, 120)
    assert out.loss.ndim == 0


def test_loss_ignores_masked_label_positions():
    llm = tiny_llm().eval()
    embeds = torch.randn(1, 6, 48)
    labels = torch.randint(0, 120, (1, 6))
    masked = labels.clone()
    masked[:, :3] = -100
    assert not torch.isclose(llm(embeds, labels=labels).loss, llm(embeds, labels=masked).loss)


def test_gradients_reach_projector_through_frozen_llm():
    llm = tiny_llm()
    projector = MLPProjector(32, llm.hidden_size)
    visual = projector(torch.randn(2, 4, 32))
    out = llm(visual, labels=torch.randint(0, 120, (2, 4)))
    out.loss.backward()
    assert projector.net[0].weight.grad is not None
    assert all(p.grad is None for p in llm.parameters())


def test_generate_returns_new_tokens_only():
    llm = tiny_llm().eval()
    out = llm.generate(torch.randn(2, 5, 48), torch.ones(2, 5, dtype=torch.long), max_new_tokens=3)
    assert out.shape == (2, 3)


@pytest.mark.network
@pytest.mark.slow
def test_real_qwen3_smoke_model_hidden_size():
    assert Qwen3Wrapper.from_pretrained(QWEN3_SMOKE).hidden_size == 1024
