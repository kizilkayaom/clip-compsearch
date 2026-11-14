from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _caption_rows(
    text_features: torch.Tensor,
    caption_image_ids: Sequence[int],
    image_ids: Sequence[int],
) -> torch.Tensor:
    if len(caption_image_ids) != len(text_features):
        raise ValueError(
            f"{len(caption_image_ids)} caption image IDs "
            f"but {len(text_features)} text rows"
        )
    row_of = {image_id: row for row, image_id in enumerate(image_ids)}
    unknown = sorted(set(caption_image_ids) - row_of.keys())
    if unknown:
        raise ValueError(f"{len(unknown)} caption image IDs not in the gallery")
    return torch.tensor([row_of[image_id] for image_id in caption_image_ids])


def image_ranks(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    caption_image_ids: Sequence[int],
    image_ids: Sequence[int],
    chunk_size: int = 1024,
) -> torch.Tensor:
    caption_rows = _caption_rows(text_features, caption_image_ids, image_ids)
    texts = F.normalize(text_features.float(), dim=-1)
    images = F.normalize(image_features.float(), dim=-1)
    ranks = []
    for start in range(0, len(images), chunk_size):
        scores = images[start : start + chunk_size] @ texts.T
        rows = torch.arange(start, start + len(scores))[:, None]
        is_true = caption_rows[None, :] == rows
        best = scores.masked_fill(~is_true, float("-inf")).max(dim=1).values
        ranks.append(((scores >= best[:, None]) & ~is_true).sum(dim=1))
    return torch.cat(ranks) if ranks else torch.empty(0, dtype=torch.long)


def caption_ranks(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    caption_image_ids: Sequence[int],
    image_ids: Sequence[int],
    chunk_size: int = 4096,
) -> torch.Tensor:
    targets = _caption_rows(text_features, caption_image_ids, image_ids)
    texts = F.normalize(text_features.float(), dim=-1)
    images = F.normalize(image_features.float(), dim=-1)
    ranks = []
    for start in range(0, len(texts), chunk_size):
        scores = texts[start : start + chunk_size] @ images.T
        true_scores = scores.gather(1, targets[start : start + chunk_size, None])
        ranks.append((scores >= true_scores).sum(dim=1) - 1)
    return torch.cat(ranks) if ranks else torch.empty(0, dtype=torch.long)


def recall_at_k(
    ranks: torch.Tensor, ks: Sequence[int] = (1, 5, 10)
) -> dict[int, float]:
    return {k: int((ranks < k).sum()) / len(ranks) for k in ks}
