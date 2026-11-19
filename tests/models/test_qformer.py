import pytest
import torch
from transformers import BertConfig, BertModel

from mlis.models.qformer import QFormer, QFormerConfig

TINY = QFormerConfig(
    num_queries=8,
    hidden_size=32,
    num_layers=4,
    num_heads=4,
    intermediate_size=64,
    vision_dim=48,
    embed_dim=16,
    vocab_size=100,
    max_text_length=16,
)


def test_default_config_is_the_planned_shape():
    config = QFormerConfig()
    assert (config.num_queries, config.hidden_size, config.num_layers) == (64, 768, 12)


def test_image_path_output_shapes():
    out = QFormer(TINY).encode_image(torch.randn(3, 20, 48))
    assert out.query_tokens.shape == (3, 8, 32)
    assert out.image_embed.shape == (3, 16)
    assert out.text_embed is None
    assert torch.allclose(out.image_embed.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_forward_with_text_returns_both_embeddings():
    model = QFormer(TINY)
    ids = torch.randint(0, 100, (3, 10))
    out = model(torch.randn(3, 20, 48), ids, torch.ones(3, 10, dtype=torch.long))
    assert out.text_embed.shape == (3, 16)
    assert torch.allclose(out.text_embed.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_default_size_has_64_by_768_queries():
    model = QFormer(QFormerConfig(num_layers=2))
    out = model.encode_image(torch.randn(1, 5, 1024))
    assert out.query_tokens.shape == (1, 64, 768)


def test_cross_attention_only_on_every_other_layer():
    model = QFormer(TINY)
    assert [layer.crossattention is not None for layer in model.layers] == [
        True,
        False,
        True,
        False,
    ]


def test_padding_does_not_change_text_embedding():
    model = QFormer(TINY).eval()
    ids = torch.randint(1, 100, (1, 6))
    padded = torch.cat([ids, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    mask = torch.tensor([[1] * 6 + [0] * 4])
    assert torch.allclose(model.encode_text(ids), model.encode_text(padded, mask), atol=1e-5)


def test_image_embedding_depends_on_image():
    model = QFormer(TINY).eval()
    a = model.encode_image(torch.randn(1, 20, 48)).image_embed
    b = model.encode_image(torch.randn(1, 20, 48)).image_embed
    assert not torch.allclose(a, b)


def test_text_longer_than_position_table_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        QFormer(TINY).encode_text(torch.zeros(1, 17, dtype=torch.long))


def test_gradients_reach_queries_and_projection():
    model = QFormer(TINY)
    out = model(torch.randn(2, 10, 48), torch.randint(0, 100, (2, 5)))
    (out.image_embed @ out.text_embed.T).sum().backward()
    assert model.query_tokens.grad is not None
    assert model.image_proj.weight.grad is not None
    assert model.layers[0].crossattention.key.weight.grad is not None


def bert_for(config: QFormerConfig) -> BertModel:
    return BertModel(
        BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_text_length,
        ),
        add_pooling_layer=False,
    )


def test_bert_initialisation_copies_weights_and_reports_untouched_ones():
    bert = bert_for(TINY)
    model = QFormer(TINY)
    untouched = model.load_bert_weights(bert.state_dict())
    state = model.state_dict()
    bert_state = bert.state_dict()
    assert torch.equal(
        state["word_embeddings.weight"], bert_state["embeddings.word_embeddings.weight"]
    )
    assert torch.equal(
        state["layers.2.attention.key.weight"],
        bert_state["encoder.layer.2.attention.self.key.weight"],
    )
    assert torch.equal(
        state["layers.1.ffn_query.intermediate.weight"],
        bert_state["encoder.layer.1.intermediate.dense.weight"],
    )
    assert torch.equal(
        state["layers.1.ffn_text.output.weight"], bert_state["encoder.layer.1.output.dense.weight"]
    )
    assert "query_tokens" in untouched
    assert any("crossattention" in name for name in untouched)
    assert "image_proj.weight" in untouched
    assert not any(name.startswith("layers.0.attention") for name in untouched)


def test_bert_initialisation_rejects_mismatched_shapes():
    other = QFormerConfig(**{**TINY.model_dump(), "hidden_size": 64})
    with pytest.raises(ValueError):
        QFormer(TINY).load_bert_weights(bert_for(other).state_dict())
