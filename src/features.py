import torch
from transformers import BatchEncoding, CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
EMBED_DIM = 512
MAX_TEXT_TOKENS = 77


def load_clip(device: str = "cpu") -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(MODEL_NAME).eval().to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


@torch.no_grad()
def image_features(model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    return model.get_image_features(pixel_values=pixel_values).pooler_output


def tokenize(processor: CLIPProcessor, texts: list[str]) -> BatchEncoding:
    return processor.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_TOKENS,
        return_tensors="pt",
    )


def text_features(
    model: CLIPModel, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    output = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    return output.pooler_output
