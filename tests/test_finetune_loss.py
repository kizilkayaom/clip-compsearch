import torch
import torch.nn.functional as F

from src.finetune.loss import contrastive_loss, similarity_logits

SCALE = 100.0


def _features(n: int, seed: int) -> torch.Tensor:
    return torch.randn(n, 512, generator=torch.Generator().manual_seed(seed))


def test_similarity_logits_shapes_for_both_arms() -> None:
    images, captions, negatives = _features(4, 0), _features(4, 1), _features(4, 2)

    i2t_b, t2i_b = similarity_logits(images, captions, None, SCALE)
    i2t_c, t2i_c = similarity_logits(images, captions, negatives, SCALE)

    assert i2t_b.shape == (4, 4)
    assert t2i_b.shape == (4, 4)
    assert i2t_c.shape == (4, 8)
    assert t2i_c.shape == (4, 4)


def test_contrastive_loss_is_mean_of_both_directions() -> None:
    images, captions = _features(4, 0), _features(4, 1)
    sims = SCALE * F.normalize(images, dim=-1) @ F.normalize(captions, dim=-1).T
    targets = torch.arange(4)

    expected = (F.cross_entropy(sims, targets) + F.cross_entropy(sims.T, targets)) / 2

    assert torch.allclose(contrastive_loss(images, captions, None, SCALE), expected)


def test_aligned_pairs_have_lower_loss_than_shuffled_pairs() -> None:
    images = _features(8, 0)
    captions = images + 0.5 * _features(8, 1)

    aligned = contrastive_loss(images, captions, None, SCALE)
    shuffled = contrastive_loss(images, captions[torch.randperm(8)], None, SCALE)

    assert aligned < shuffled


def test_hard_negatives_are_extra_image_to_text_columns() -> None:
    images = _features(4, 0)
    captions = images + 0.5 * _features(4, 1)
    far = -captions
    close = captions + 0.01 * _features(4, 2)

    no_negatives = contrastive_loss(images, captions, None, SCALE)
    far_negatives = contrastive_loss(images, captions, far, SCALE)
    close_negatives = contrastive_loss(images, captions, close, SCALE)

    assert torch.allclose(far_negatives, no_negatives, atol=1e-4)
    assert close_negatives > no_negatives + 0.1
