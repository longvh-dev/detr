import pytest
import torch
from detr.modules.transformers import Transformer


@pytest.fixture
def transformer_configs(request):
    """Fixture providing different transformer configurations for testing"""
    return request.param


@pytest.fixture
def sample_inputs(request):
    """Fixture providing sample input tensors for transformer testing"""
    batch_size = request.param["batch_size"]
    d_model = request.param["d_model"]

    src = torch.rand((batch_size, 3, 224, 224))
    mask = torch.zeros((batch_size, 224, 224)).bool()
    query_embed = torch.rand((100, d_model))
    pos_embed = torch.rand((batch_size, d_model, 224, 224))

    return {
        "src": src,
        "mask": mask,
        "query_embed": query_embed,
        "pos_embed": pos_embed,
    }


@pytest.mark.parametrize(
    "transformer_configs",
    [
        {"batch_size": 4, "d_model": 256, "nhead": 4},
        {"batch_size": 8, "d_model": 384, "nhead": 6},
        {"batch_size": 16, "d_model": 512, "nhead": 8},
    ],
    indirect=True,
)
def test_transformer_initialization(transformer_configs):
    """Test transformer initialization with different configurations"""
    transformer = Transformer(
        d_model=transformer_configs["d_model"], nhead=transformer_configs["nhead"]
    )

    assert transformer.encoder is not None, "Encoder should be initialized"
    assert transformer.decoder is not None, "Decoder should be initialized"
    assert transformer.d_model == transformer_configs["d_model"], (
        "d_model should match config"
    )
    assert transformer.nhead == transformer_configs["nhead"], (
        "nhead should match config"
    )


@pytest.mark.parametrize("batch_size,d_model", [(4, 256), (8, 384), (16, 512)])
def test_transformer_output_shape(batch_size, d_model):
    """Test transformer output shapes"""
    transformer = Transformer(d_model=d_model, nhead=8)

    src = torch.rand((batch_size, d_model, 224, 224))
    mask = torch.zeros((batch_size, 224, 224)).bool()
    query_embed = torch.rand((100, d_model))
    pos_embed = torch.rand((batch_size, d_model, 224, 224))

    output, memory = transformer(
        src=src, mask=mask, query_embed=query_embed, pos_embed=pos_embed
    )

    assert output.shape == (batch_size, 100, d_model)
    assert memory.shape == (batch_size, 100, d_model)
