import pytest
import torch
from transformers import CLIPModel, CLIPProcessor

from src.features import EMBED_DIM, MODEL_NAME, image_features, load_clip


@pytest.fixture(scope="module")
def clip() -> tuple[CLIPModel, CLIPProcessor]:
    return load_clip("cpu")


def test_load_clip_returns_model_in_eval_mode(
    clip: tuple[CLIPModel, CLIPProcessor],
) -> None:
    model, _ = clip

    assert not model.training
    assert model.config.name_or_path == MODEL_NAME


def test_image_features_are_projected_embeddings(
    clip: tuple[CLIPModel, CLIPProcessor],
) -> None:
    model, _ = clip
    generator = torch.Generator().manual_seed(0)
    pixel_values = torch.randn(2, 3, 224, 224, generator=generator)

    features = image_features(model, pixel_values)

    with torch.no_grad():
        pooled = model.vision_model(pixel_values=pixel_values).pooler_output
        expected = model.visual_projection(pooled)
    assert features.shape == (2, EMBED_DIM)
    assert torch.allclose(features, expected, atol=1e-6)
    assert not features.requires_grad
