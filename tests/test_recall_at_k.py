import pytest
import torch

from src.eval.recall_at_k import caption_ranks, image_ranks, recall_at_k

IMAGE_IDS = [10, 20, 30]
IMAGES = torch.eye(3)
CAPTION_IMAGE_IDS = [10, 10, 20, 20, 30, 30]
TEXTS = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.5, 0.9, 0.0],
        [0.0, 1.0, 0.0],
        [0.6, 0.1, 0.9],
        [0.0, 0.0, 1.0],
        [0.9, 0.0, 0.1],
    ]
)
EXPECTED_RANKS = [0, 1, 0, 2, 0, 1]


def test_caption_ranks_counts_better_images() -> None:
    ranks = caption_ranks(TEXTS, IMAGES, CAPTION_IMAGE_IDS, IMAGE_IDS)

    assert ranks.tolist() == EXPECTED_RANKS


def test_recall_at_k_hits_at_rank_k_and_misses_at_rank_k_plus_one() -> None:
    ranks = torch.tensor([1, 2])

    assert recall_at_k(ranks, (2,)) == {2: 0.5}


def test_recall_at_k_on_toy_data() -> None:
    ranks = caption_ranks(TEXTS, IMAGES, CAPTION_IMAGE_IDS, IMAGE_IDS)

    assert recall_at_k(ranks, (1, 2, 3)) == {1: 0.5, 2: 5 / 6, 3: 1.0}


def test_caption_ranks_use_image_ids_not_row_positions() -> None:
    order = [2, 0, 1]
    shuffled_ids = [IMAGE_IDS[row] for row in order]

    ranks = caption_ranks(TEXTS, IMAGES[order], CAPTION_IMAGE_IDS, shuffled_ids)

    assert ranks.tolist() == EXPECTED_RANKS


def test_caption_ranks_count_ties_against_the_true_image() -> None:
    texts = torch.tensor([[1.0, 1.0, 0.0]])

    assert caption_ranks(texts, IMAGES, [10], IMAGE_IDS).tolist() == [1]


def test_caption_ranks_ignore_feature_scale_and_chunk_size() -> None:
    scaled = caption_ranks(TEXTS * 7, IMAGES * 3, CAPTION_IMAGE_IDS, IMAGE_IDS)
    chunked = caption_ranks(TEXTS, IMAGES, CAPTION_IMAGE_IDS, IMAGE_IDS, chunk_size=1)

    assert scaled.tolist() == EXPECTED_RANKS
    assert chunked.tolist() == EXPECTED_RANKS


def test_caption_ranks_reject_unknown_image_id() -> None:
    with pytest.raises(ValueError):
        caption_ranks(TEXTS[:1], IMAGES, [99], IMAGE_IDS)


def test_caption_ranks_reject_length_mismatch() -> None:
    with pytest.raises(ValueError):
        caption_ranks(TEXTS, IMAGES, CAPTION_IMAGE_IDS[:2], IMAGE_IDS)


def test_image_ranks_use_the_best_ranked_caption_of_each_image() -> None:
    ranks = image_ranks(TEXTS, IMAGES, CAPTION_IMAGE_IDS, IMAGE_IDS)

    assert ranks.tolist() == [0, 0, 0]


def test_image_ranks_count_other_images_captions_above_the_best_true_one() -> None:
    texts = torch.tensor([[0.6, 0.0, 0.8], [0.0, 0.6, 0.8], [0.8, 0.0, 0.6]])

    ranks = image_ranks(texts, IMAGES, [10, 20, 30], IMAGE_IDS)

    assert ranks.tolist() == [1, 0, 2]


def test_image_ranks_ignore_order_and_chunk_size() -> None:
    order = [2, 0, 1]
    shuffled_ids = [IMAGE_IDS[row] for row in order]

    ranks = image_ranks(
        TEXTS, IMAGES[order], CAPTION_IMAGE_IDS, shuffled_ids, chunk_size=1
    )

    assert ranks.tolist() == [0, 0, 0]
