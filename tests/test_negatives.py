"""Tests for the caption rebuild in src/data/negatives.py."""

from typing import Callable

import pytest

from src.data.negatives import (
    generate_negative,
    get_nli_model,
    negative_from_doc,
    nli_filter,
    nli_keep,
    nli_scores,
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
    pytest.param(
        swap_objects,
        "A tennis player with a racket.",
        "A racket with a tennis player.",
        id="objects-compound_unit",
    ),
    pytest.param(
        swap_objects,
        "a baseball player on the home plate during a game",
        "a home plate on the baseball player during a game",
        id="objects-two_compound_units",
    ),
    pytest.param(
        swap_objects,
        "A dining room table with a vase.",
        "A vase with a dining room table.",
        id="objects-compound_chain",
    ),
    pytest.param(
        swap_objects,
        "A woman with an umbrella near the sea.",
        "An umbrella with a woman near the sea.",
        id="objects-article_a_an",
    ),
    pytest.param(
        swap_objects,
        "A man in a uniform holds a bat.",
        "A uniform in a man holds a bat.",
        id="objects-article_exception",
    ),
    pytest.param(
        swap_objects,
        "A group of kids on a field",
        "A field on a group of kids",
        id="objects-x_of_y_unit",
    ),
    pytest.param(
        swap_objects,
        "A motorcycle parked in front of a cafe",
        "A cafe parked in front of a motorcycle",
        id="objects-position_noun_skipped",
    ),
    pytest.param(
        swap_objects,
        "A slice of a piece of cake on a plate",
        "A plate on a slice of a piece of cake",
        id="objects-nested_x_of_y",
    ),
    pytest.param(
        swap_objects,
        "A living room with a couch, coffee table and two large windows.",
        "A couch with a living room, coffee table and two large windows.",
        id="objects-amod_noun_modifier",
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


@pytest.mark.xfail(
    strict=True,
    reason="Known limit: NLI scores the coordination swap as contradiction",
)
def test_nli_filter_rejects_coordination_swap() -> None:
    caption = "A dog and a cat sit on a sofa."
    assert not nli_filter(caption, "A cat and a dog sit on a sofa.")


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        pytest.param(
            "The cat chased the dog.",
            ("The dog chased the cat.", "noun"),
            id="noun",
        ),
        pytest.param(
            "A small red car.",
            ("A red small car.", "adjective"),
            id="adjective",
        ),
    ],
)
def test_negative_from_doc_reports_swap_type(
    caption: str, expected: tuple[str, str]
) -> None:
    negative = negative_from_doc(nlp(caption))

    assert negative is not None
    assert (negative.text, negative.swap_type) == expected


@pytest.mark.parametrize(
    ("caption", "relation"),
    [
        pytest.param("A dog and a cat sit on a sofa.", "and", id="and"),
        pytest.param("A cat or a dog sleeps on a bed.", "or", id="or"),
        pytest.param(
            "Meat, vegetables, fruit, and bread on a plate.", "and", id="list"
        ),
        pytest.param("A cat sitting next to a dog.", "next to", id="next_to"),
        pytest.param("A bench beside a tree.", "beside", id="beside"),
        pytest.param("A car parked near a tree.", "near", id="near"),
        pytest.param("A bench by a lake.", "by", id="by"),
        pytest.param("A path alongside a river.", "alongside", id="alongside"),
        pytest.param("A store across from a park.", "across from", id="across_from"),
        pytest.param("A man sitting opposite a woman.", "opposite", id="opposite"),
        pytest.param("A woman with an umbrella near the sea.", None, id="umbrella"),
        pytest.param("A photo taken by a man.", None, id="agent_by"),
        pytest.param("A man with a dog and a cat.", None, id="and_elsewhere"),
    ],
)
def test_negative_from_doc_reports_symmetric_relation(
    caption: str, relation: str | None
) -> None:
    negative = negative_from_doc(nlp(caption))

    assert negative is not None
    assert negative.relation == relation


def test_negative_from_doc_returns_none_without_swap() -> None:
    assert negative_from_doc(nlp("A dog is running.")) is None


def test_nli_keep_scores_a_batch() -> None:
    pairs = [
        ("A dog runs on the grass.", "A dog runs on the grass."),
        ("The cat chased the dog.", "The dog chased the cat."),
    ]

    assert nli_keep(pairs) == [False, True]


def test_nli_keep_accepts_empty_batch() -> None:
    assert nli_keep([]) == []


def test_nli_scores_returns_three_scores_per_pair() -> None:
    pairs = [("A dog runs on the grass.", "A dog runs on the grass.")]

    scores = nli_scores(pairs)

    assert len(scores) == 1
    assert len(scores[0]) == 3
    assert max(range(3), key=lambda label: scores[0][label]) == 1


def test_nli_scores_accepts_empty_batch() -> None:
    assert nli_scores([]) == []
