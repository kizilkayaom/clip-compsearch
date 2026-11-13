from dataclasses import dataclass
from functools import cache

import numpy as np
import spacy
from sentence_transformers import CrossEncoder
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
ENTAILMENT = 1
MODIFIER_DEPS = frozenset({"amod", "compound", "nmod"})
NOUN_POS = frozenset({"NOUN", "PROPN"})
POSITION_NOUNS = frozenset(
    {
        "front", "top", "side", "middle", "back",
        "edge", "bottom", "end", "corner", "center",
    }
)
SWAP_TYPES = {"NOUN": "noun", "ADJ": "adjective"}
COORDINATORS = frozenset({"and", "or"})
SYMMETRIC_RELATIONS = frozenset(
    {"next to", "beside", "near", "by", "alongside", "across from", "opposite"}
)
TWO_WORD_PREPOSITIONS = frozenset({("next", "to"), ("across", "from")})
ARTICLES = frozenset({"a", "an"})
VOWELS = frozenset("aeiou")
A_BEFORE_VOWEL = ("uni", "use", "usu", "eu", "one", "once")
AN_BEFORE_CONSONANT = ("hour", "honest", "honor", "heir")

nlp = spacy.load("en_core_web_sm")


@dataclass(frozen=True)
class Negative:
    text: str
    swap_type: str
    relation: str | None


@cache
def get_nli_model() -> CrossEncoder:
    return CrossEncoder(NLI_MODEL_NAME)


def _recase(word: Token, target: Token) -> str:
    text = word.text
    if target.is_sent_start:
        return text[:1].upper() + text[1:]
    if word.is_sent_start and not text.isupper():
        return text[:1].lower() + text[1:]
    return text


def _article_for(word: str) -> str:
    lower = word.lower()
    if lower.startswith(AN_BEFORE_CONSONANT):
        return "an"
    if lower.startswith(A_BEFORE_VOWEL) or lower[:1] not in VOWELS:
        return "a"
    return "an"


def _match_article(article: str, word: str) -> str:
    target = _article_for(word)
    if len(article) > 1 and article.isupper():
        return target.upper()
    if article[:1].isupper():
        return target.capitalize()
    return target


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


def _is_noun_modifier(token: Token) -> bool:
    return (
        token.pos_ in NOUN_POS
        and token.dep_ in MODIFIER_DEPS
        and token.head.pos_ in NOUN_POS
        and token.head.i > token.i
    )


def _compound_start(doc: Doc, head: Token) -> int:
    start = head.i
    while start > 0:
        modifier = doc[start - 1]
        if not _is_noun_modifier(modifier):
            break
        if not start <= modifier.head.i <= head.i:
            break
        start -= 1
    return start


def _merge_compounds(doc: Doc) -> Doc:
    spans = []
    for token in doc:
        if token.pos_ in NOUN_POS and not _is_noun_modifier(token):
            start = _compound_start(doc, token)
            if start < token.i:
                spans.append(doc[start : token.i + 1])

    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)
    return doc


def _is_position_noun(token: Token) -> bool:
    doc = token.doc
    return (
        token.lower_ in POSITION_NOUNS
        and token.i + 1 < len(doc)
        and doc[token.i + 1].lower_ == "of"
    )


def _of_object(token: Token) -> Token | None:
    doc = token.doc
    if token.i + 1 >= len(doc):
        return None
    preposition = doc[token.i + 1]
    if preposition.lower_ != "of" or preposition.head.i != token.i:
        return None
    for child in preposition.children:
        if child.dep_ == "pobj" and child.pos_ in NOUN_POS:
            return child
    return None


def _merge_of_phrases(doc: Doc) -> Doc:
    spans = []
    covered_until = -1
    for token in doc:
        if token.i <= covered_until or token.pos_ not in NOUN_POS:
            continue
        if _is_position_noun(token):
            continue
        end = token
        inner = _of_object(end)
        while inner is not None:
            end = inner
            inner = _of_object(end)
        if end.i > token.i:
            spans.append(doc[token.i : end.i + 1])
            covered_until = end.i

    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)
    return doc


def _merge_units(doc: Doc) -> Doc:
    return _merge_of_phrases(_merge_compounds(_merge_hyphenated(doc)))


def _coordinator(first: Token, second: Token) -> str | None:
    if second.i not in {token.i for token in first.conjuncts}:
        return None
    for member in (first, *first.conjuncts):
        for child in member.children:
            if child.dep_ == "cc" and child.lower_ in COORDINATORS:
                return child.lower_
    return None


def _preposition_phrase(preposition: Token) -> str:
    if preposition.i == 0:
        return preposition.lower_
    previous = preposition.doc[preposition.i - 1].lower_
    if (previous, preposition.lower_) in TWO_WORD_PREPOSITIONS:
        return f"{previous} {preposition.lower_}"
    return preposition.lower_


def _symmetric_preposition(first: Token, second: Token) -> str | None:
    preposition = second.head
    if second.dep_ != "pobj" or preposition.dep_ == "agent":
        return None
    if not first.i < preposition.i < second.i:
        return None
    phrase = _preposition_phrase(preposition)
    return phrase if phrase in SYMMETRIC_RELATIONS else None


def _relation(first: Token, second: Token) -> str | None:
    return _coordinator(first, second) or _symmetric_preposition(first, second)


def _swap_pair(doc: Doc, pos: str) -> tuple[Token, Token] | None:
    tokens = [
        token for token in doc if token.pos_ == pos and not _is_position_noun(token)
    ]
    others = [token for token in tokens[1:] if token.lower_ != tokens[0].lower_]
    if not others:
        return None
    return tokens[0], others[0]


def _render_swap(doc: Doc, first: Token, second: Token) -> str:
    words = [token.text for token in doc]
    words[first.i] = _recase(second, first)
    words[second.i] = _recase(first, second)
    for unit in (first, second):
        if unit.i > 0 and doc[unit.i - 1].lower_ in ARTICLES:
            article = doc[unit.i - 1].text
            words[unit.i - 1] = _match_article(article, words[unit.i])
    return "".join(word + token.whitespace_ for word, token in zip(words, doc))


def _swap_in_doc(doc: Doc, pos: str) -> Negative | None:
    pair = _swap_pair(doc, pos)
    if pair is None:
        return None
    first, second = pair
    text = _render_swap(doc, first, second)
    return Negative(text, SWAP_TYPES[pos], _relation(first, second))


def negative_from_doc(doc: Doc) -> Negative | None:
    doc = _merge_units(doc)
    for pos in SWAP_TYPES:
        negative = _swap_in_doc(doc, pos)
        if negative is not None:
            return negative
    return None


def _swap_first_two_distinct(nlp: Language, caption: str, pos: str) -> tuple[str, bool]:
    negative = _swap_in_doc(_merge_units(nlp(caption)), pos)
    return (caption, False) if negative is None else (negative.text, True)


def swap_objects(nlp: Language, caption: str) -> tuple[str, bool]:
    return _swap_first_two_distinct(nlp, caption, "NOUN")


def swap_attributes(nlp: Language, caption: str) -> tuple[str, bool]:
    return _swap_first_two_distinct(nlp, caption, "ADJ")


def generate_negative(nlp: Language, caption: str) -> tuple[str, bool]:
    negative = negative_from_doc(nlp(caption))
    return (caption, False) if negative is None else (negative.text, True)


def nli_scores(
    pairs: list[tuple[str, str]], batch_size: int = 256
) -> list[list[float]]:
    if not pairs:
        return []
    scores = get_nli_model().predict(pairs, batch_size=batch_size)
    return [[float(value) for value in row] for row in scores]


def nli_keep(pairs: list[tuple[str, str]], batch_size: int = 256) -> list[bool]:
    scores = nli_scores(pairs, batch_size=batch_size)
    return [int(np.argmax(row)) != ENTAILMENT for row in scores]


def nli_filter(original: str, negative: str) -> bool:
    return nli_keep([(original, negative)])[0]
