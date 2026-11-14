from collections.abc import Sequence

import torch
import torch.nn.functional as F


def caption_ranks(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    caption_image_ids: Sequence[int],
    image_ids: Sequence[int],
    chunk_size: int = 4096,
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

    targets = torch.tensor([row_of[image_id] for image_id in caption_image_ids])
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
