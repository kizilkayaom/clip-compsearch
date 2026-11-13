"""Tests for the caption rebuild in src/data/negatives.py."""

from typing import Callable

import pytest

from src.data.negatives import (
    generate_negative,
    get_nli_model,
    nli_filter,
    nlp,
    swap_attributes,
    swap_objects,
)

SwapFn = Callable[..., tuple[str, bool]]

NO_SWAP_CASES = [
    pytest.param(
        swap_objects,
        "NOUN",
        "A dog is running.",
        ["dog"],
        id="objects-one_noun",
    ),
    pytest.param(
        swap_objects,
        "NOUN",
        "A dog chases a dog.",
        ["dog", "dog"],
        id="objects-same_nouns",
    ),
    pytest.param(
        swap_attributes,
        "ADJ",
        "A dog is running.",
        [],
        id="attributes-no_adj",
    ),
    pytest.param(
        swap_attributes,
        "ADJ",
        "A red car and a red bus.",
        ["red", "red"],
        id="attributes-same_adj",
    ),
]

SWAP_CASES = [
    pytest.param(
        swap_objects,
        "The cat chased the dog.",
        "The dog chased the cat.",
        id="objects-trailing_period",
    ),
    pytest.param(
        swap_objects,
        "A dog's bowl is on the floor.",
        "A bowl's dog is on the floor.",
        id="objects-apostrophe",
    ),
    pytest.param(
        swap_objects,
        "A man wears a t-shirt.",
        "A t-shirt wears a man.",
        id="objects-hyphen",
    ),
    pytest.param(
        swap_objects,
        "A cat  sits on a mat.",
        "A mat  sits on a cat.",
        id="objects-double_space",
    ),
    pytest.param(
        swap_objects,
        "Cats sleep on a sofa.",
        "Sofa sleep on a cats.",
        id="objects-capitalized",
    ),
    pytest.param(
        swap_objects,
        "T-shirt hangs on a hook.",
        "Hook hangs on a t-shirt.",
        id="objects-leaves_sentence_start",
    ),
    pytest.param(
        swap_objects,
        "A TV sits on a table.",
        "A table sits on a TV.",
        id="objects-acronym",
    ),
    pytest.param(
        swap_objects,
        "A two-story house with a red door.",
        "A two-story door with a red house.",
        id="objects-hyphen_modifier",
    ),
    pytest.param(
        swap_objects,
        "A man rides a horse on a beach.",
        "A horse rides a man on a beach.",
        id="objects-first_two_of_three",
    ),
    pytest.param(
        swap_objects,
        "A dog and a cat chase a dog.",
        "A cat and a dog chase a dog.",
        id="objects-skip_repeated_noun",
    ),
    pytest.param(
        swap_attributes,
        "A red mug sits on a blue book.",
        "A blue mug sits on a red book.",
        id="attributes-trailing_period",
    ),
    pytest.param(
        swap_attributes,
        "A two-story house with a red door.",
        "A red house with a two-story door.",
        id="attributes-hyphen_modifier",
    ),
]

GENERATE_CASES = [
    pytest.param(
        "A dog runs on green grass under a blue sky.",
        ("A grass runs on green dog under a blue sky.", True),
        id="noun_swap_first",
    ),
    pytest.param(
        "A small red car.",
        ("A red small car.", True),
        id="adj_fallback-one_noun",
    ),
    pytest.param(
        "A red dog chased a brown dog.",
        ("A brown dog chased a red dog.", True),
        id="adj_fallback-same_nouns",
    ),
    pytest.param("A dog is running.", ("A dog is running.", False), id="no_swap"),
]


def _words_with_pos(caption: str, pos: str) -> list[str]:
    return [token.text for token in nlp(caption) if token.pos_ == pos]


@pytest.mark.parametrize(("swap_fn", "pos", "caption", "pos_words"), NO_SWAP_CASES)
def test_no_swap_returns_caption_and_false(
    swap_fn: SwapFn, pos: str, caption: str, pos_words: list[str]
) -> None:
    assert _words_with_pos(caption, pos) == pos_words, "spaCy tags changed"

    assert swap_fn(nlp, caption) == (caption, False)


@pytest.mark.parametrize(("swap_fn", "caption", "expected"), SWAP_CASES)
def test_swap_changes_only_the_two_swapped_words(
    swap_fn: SwapFn, caption: str, expected: str
) -> None:
    assert swap_fn(nlp, caption) == (expected, True)


@pytest.mark.parametrize(("caption", "expected"), GENERATE_CASES)
def test_generate_negative_tries_nouns_then_adjectives(
    caption: str, expected: tuple[str, bool]
) -> None:
    assert generate_negative(nlp, caption) == expected


def test_nli_label_1_is_entailment() -> None:
    assert get_nli_model().config.id2label[1] == "entailment"


def test_nli_filter_rejects_identical_caption() -> None:
    assert not nli_filter("A dog runs on the grass.", "A dog runs on the grass.")


def test_nli_filter_keeps_swapped_caption() -> None:
    assert nli_filter("The cat chased the dog.", "The dog chased the cat.")
