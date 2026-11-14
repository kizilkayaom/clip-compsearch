import torch

from demo.app import rank_gallery, verdict
from src.eval.validation import encode_texts
from src.features import load_clip

CAPTIONS = ["A dog runs on the grass.", "A red car on a street.", "A plate of pasta."]


def test_rank_gallery_orders_images_by_cosine() -> None:
    gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    query = torch.tensor([[2.0, 0.2]])

    ids, scores = rank_gallery(query, gallery, [10, 20, 30], k=2)

    assert ids == [10, 30]
    assert scores[0] > scores[1]


def test_rank_gallery_finds_the_matching_text_embedding() -> None:
    model, processor = load_clip("cpu")
    gallery = encode_texts(model, processor, CAPTIONS, batch_size=3)
    query = encode_texts(model, processor, [CAPTIONS[1]], batch_size=1)

    ids, _ = rank_gallery(query, gallery, [10, 20, 30], k=3)

    assert ids[0] == 20


def test_verdict_passes_only_when_the_caption_wins() -> None:
    assert verdict(0.31, 0.29) == "Pass"
    assert verdict(0.29, 0.31) == "Fail"
    assert verdict(0.30, 0.30) == "Fail"
