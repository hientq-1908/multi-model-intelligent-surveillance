import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from mlis.data.reader import iter_samples
from mlis.models.image_encoder import RandomImageEncoder
from mlis.models.preprocess import PERSON_SIZE, ImagePreprocessor
from mlis.models.projector import MLPProjector
from mlis.models.qformer import QFormer, QFormerConfig
from mlis.models.qwen3 import Qwen3Wrapper


def test_fixture_images_flow_through_encoder_qformer_projector_and_llm(shard_manifest):
    images = [s.image for s in iter_samples(shard_manifest)][:4]
    pixels = ImagePreprocessor(PERSON_SIZE)(images)
    encoder = RandomImageEncoder(output_dim=24)
    qformer = QFormer(
        QFormerConfig(
            num_queries=8,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            intermediate_size=64,
            vision_dim=24,
            embed_dim=16,
        )
    )
    llm = Qwen3Wrapper(
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=120,
                hidden_size=48,
                intermediate_size=96,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=12,
            )
        )
    )
    projector = MLPProjector(qformer.hidden_size, llm.hidden_size)

    visual = projector(qformer.encode_image(encoder(pixels)).query_tokens)
    text = llm.embed_tokens(torch.randint(0, 120, (4, 5)))
    out = llm(torch.cat([visual, text], dim=1))

    assert visual.shape == (4, 8, 48)
    assert out.logits.shape == (4, 8 + 5, 120)
