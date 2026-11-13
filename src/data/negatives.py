from functools import cache

import numpy as np
import spacy
from sentence_transformers import CrossEncoder
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

MODIFIER_DEPS = frozenset({"amod", "compound", "nmod"})
NOUN_POS = frozenset({"NOUN", "PROPN"})

nlp = spacy.load("en_core_web_sm")


@cache
def get_nli_model() -> CrossEncoder:
    return CrossEncoder("cross-encoder/nli-deberta-v3-small")


def _recase(word: Token, target: Token) -> str:
    text = word.text
    if target.is_sent_start:
        return text[:1].upper() + text[1:]
    if word.is_sent_start and not text.isupper():
        return text[:1].lower() + text[1:]
    return text


def _merged_pos(span: Span) -> str:
    root = span.root
    if root.dep_ in MODIFIER_DEPS and root.head.pos_ in NOUN_POS:
        return "ADJ"
    return root.pos_


def _merge_hyphenated(doc: Doc) -> Doc:
    spans = []
    start = 0
    while start < len(doc) - 2:
        end = start
        while (
            end + 2 < len(doc)
            and doc[end].whitespace_ == ""
            and doc[end + 1].text == "-"
            and doc[end + 1].whitespace_ == ""
        ):
            end += 2
        if end > start:
            spans.append(doc[start : end + 1])
        start = end + 1

    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span, attrs={"POS": _merged_pos(span)})
    return doc


def _swap_first_two_distinct(nlp: Language, caption: str, pos: str) -> tuple[str, bool]:
    doc = _merge_hyphenated(nlp(caption))
    tokens = [token for token in doc if token.pos_ == pos]
    others = [token for token in tokens[1:] if token.lower_ != tokens[0].lower_]
    if not others:
        return caption, False

    first, second = tokens[0], others[0]
    words = [token.text for token in doc]
    words[first.i] = _recase(second, first)
    words[second.i] = _recase(first, second)
    negative = "".join(word + token.whitespace_ for word, token in zip(words, doc))
    return negative, True


def swap_objects(nlp: Language, caption: str) -> tuple[str, bool]:
    return _swap_first_two_distinct(nlp, caption, "NOUN")


def swap_attributes(nlp: Language, caption: str) -> tuple[str, bool]:
    return _swap_first_two_distinct(nlp, caption, "ADJ")


def generate_negative(nlp: Language, caption: str) -> tuple[str, bool]:
    negative, swapped = swap_objects(nlp, caption)
    if swapped:
        return negative, swapped
    return swap_attributes(nlp, caption)


def nli_filter(original: str, negative: str) -> bool:
    scores = get_nli_model().predict((original, negative))
    if np.argmax(scores) == 1:
        return False
    return True
