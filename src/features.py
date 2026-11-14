import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
EMBED_DIM = 512


def load_clip(device: str = "cpu") -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(MODEL_NAME).eval().to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


@torch.no_grad()
def image_features(model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    return model.get_image_features(pixel_values=pixel_values).pooler_output
