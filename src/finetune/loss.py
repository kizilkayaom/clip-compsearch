import torch
import torch.nn.functional as F


def similarity_logits(
    image_features: torch.Tensor,
    caption_features: torch.Tensor,
    negative_features: torch.Tensor | None,
    logit_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = F.normalize(image_features, dim=-1)
    captions = F.normalize(caption_features, dim=-1)
    texts = captions
    if negative_features is not None:
        texts = torch.cat([captions, F.normalize(negative_features, dim=-1)])
    image_to_text = logit_scale * images @ texts.T
    text_to_image = logit_scale * captions @ images.T
    return image_to_text, text_to_image


def contrastive_loss(
    image_features: torch.Tensor,
    caption_features: torch.Tensor,
    negative_features: torch.Tensor | None,
    logit_scale: float,
) -> torch.Tensor:
    image_to_text, text_to_image = similarity_logits(
        image_features, caption_features, negative_features, logit_scale
    )
    targets = torch.arange(len(image_features), device=image_features.device)
    loss_i2t = F.cross_entropy(image_to_text, targets)
    loss_t2i = F.cross_entropy(text_to_image, targets)
    return (loss_i2t + loss_t2i) / 2
