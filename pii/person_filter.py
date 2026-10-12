"""Decide whether a PERSON span is really a person's name, using evidence from the
document itself instead of word lists.

spaCy and the contextual name pattern tag any run of capitalised words, so headings
("Board Risk Committee"), role phrases ("Head of Enterprise") and OCR noise came back
as PERSON, and spans swallowed their neighbours ("Marguerite Vossberg Led").

Three rules, all learned per document:
1. Lowercase test: a word written in lowercase a real share of the time in the document
   ("risk", "committee", "led", "head") is an ordinary word, not part of a name.
   Names are virtually never written in lowercase.
2. Email evidence: name parts in the document's own email addresses
   ("n.osadebe@..." -> "osadebe") confirm a surname, even one that is also a word.
3. Model agreement: a span found only by our regex pattern (not by spaCy's statistical
   model) needs evidence: an email-confirmed part, or a word from a name spaCy found.

Only shape rules are fixed: capitalisation, initials, name particles, and a few
grammar words used for trimming.
"""

import re
from collections import Counter
from collections.abc import Iterable

from config.settings import CORP_EMAIL_PATTERN

# Grammar words trimmed from a span's edges ("by Tobias Lindqvist", "Head of ...").
EDGE_WORDS = {"by", "of", "the", "and", "to", "at", "for", "with", "via", "from", "on", "in"}

PARTICLES = {"van", "von", "der", "den", "de", "del", "da", "di", "du", "la", "le", "bin", "ibn", "al"}

_TOKEN = re.compile(r"\S+")
_INITIAL = re.compile(r"^[A-Z]\.(?:-?[A-Z]\.)?$")
_NAME_WORD = re.compile(r"^[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*$")
_EMAIL = re.compile(CORP_EMAIL_PATTERN.regex)
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
# A word written entirely in lowercase (not part of a longer mixed-case word).
_LOWER_WORD = re.compile(r"(?<![A-Za-z])[a-z][a-z'’-]*[a-z](?![A-Za-z])")
_ANY_WORD = re.compile(r"(?<![A-Za-z])[A-Za-z][A-Za-z'’-]*[A-Za-z](?![A-Za-z])")

# A word is ordinary if it is written in lowercase more than once, or in at least a
# quarter of its uses. Names are almost never lowercase; a single lowercase sighting is
# tolerated because OCR occasionally lowercases a name ("A. sorokina").
COMMON_LOWERCASE_MIN = 2
COMMON_LOWERCASE_SHARE = 0.25


def _keys(word: str) -> list[str]:
    """Lowercased word and, for hyphenated words, its parts ("risk-based" -> risk, based)."""
    word = word.lower().replace("’", "'").removesuffix("'s")
    return [word] + [part for part in word.split("-") if len(part) > 1 and part != word]


def _letters(token: str) -> str:
    return re.sub(r"[^a-z]", "", token.lower())


class DocumentVocabulary:
    """What one document says about its own words: which are ordinary words (seen in
    lowercase) and which are confirmed name parts (seen in email addresses)."""

    def __init__(self, texts: Iterable[str]):
        self.email_names: set[str] = set()
        locals_: list[str] = []
        lower_counts: Counter = Counter()
        all_counts: Counter = Counter()
        for text in texts:
            if not text:
                continue
            locals_ += [m.group().split("@")[0].strip() for m in _EMAIL.finditer(text)]
            cleaned = _URL.sub(" ", _EMAIL.sub(" ", text))
            for word in _ANY_WORD.findall(cleaned):
                for key in _keys(word):
                    all_counts[key] += 1
            for word in _LOWER_WORD.findall(cleaned):
                for key in _keys(word):
                    lower_counts[key] += 1
        # Ordinary words are written in lowercase repeatedly ("risk", "board", "led");
        # a name is not, even if OCR lowercased it once ("sorokina").
        self.common_words: set[str] = {
            word for word, n in lower_counts.items()
            if n >= COMMON_LOWERCASE_MIN or n / max(n, all_counts[word]) >= COMMON_LOWERCASE_SHARE
        }
        for local in locals_:
            pieces = [p for p in re.split(r"[._\-+\d]+", local.lower()) if p]
            parts = [p for p in pieces if len(p) >= 3]
            if len(pieces) == 2 and len(pieces[0]) == 1 and parts:
                # initial.surname ("g.park"): a person's mailbox, so the surname is
                # confirmed even if it is also an ordinary word.
                self.email_names.update(parts)
            else:
                # Otherwise only parts that are not ordinary words: "board.chair@" is
                # a role mailbox and confirms nothing.
                self.email_names.update(p for p in parts if p not in self.common_words)

    def is_common(self, token: str) -> bool:
        lower = token.lower().replace("’", "'").removesuffix("'s")
        return lower in self.common_words and not self.is_email_confirmed(token)

    def is_email_confirmed(self, token: str) -> bool:
        letters = _letters(token)
        return len(letters) >= 3 and letters in self.email_names


def _core(token: str) -> str:
    """Token without surrounding punctuation (keeps an initial's dot)."""
    stripped = token.strip(",;:()[]{}\"'“”‘’|—–")
    if _INITIAL.match(stripped):
        return stripped
    return stripped.rstrip(".").removesuffix("'s").removesuffix("’s")


def _is_edge_token(token: str, vocab: DocumentVocabulary) -> bool:
    core = _core(token)
    if not core or not any(ch.isalpha() for ch in core):
        return True
    if vocab.is_email_confirmed(core):
        return False
    if core.lower() in EDGE_WORDS or vocab.is_common(core):
        return True
    stripped = token.strip(",;:()[]{}\"'“”‘’|—–")
    # Abbreviations with a dot that are not initials ("Dir.", "Div.").
    if stripped.endswith(".") and not _INITIAL.match(stripped) and len(core) <= 4:
        return True
    # Headings and labels in capitals ("DAYS OVERDUE", "CRO"), but not initials.
    return core.isupper() and len(core) >= 2 and not _INITIAL.match(core)


def trim_person_span(
    text: str, start: int, end: int, vocab: DocumentVocabulary
) -> tuple[int, int]:
    """Shrink [start, end) so it does not begin or end with ordinary words, headings,
    abbreviations or punctuation. Returns (start, start) if nothing is left."""
    tokens = [(m.start() + start, m.end() + start) for m in _TOKEN.finditer(text[start:end])]
    while tokens and _is_edge_token(text[tokens[0][0] : tokens[0][1]], vocab):
        tokens.pop(0)
    while tokens and _is_edge_token(text[tokens[-1][0] : tokens[-1][1]], vocab):
        tokens.pop()
    if not tokens:
        return start, start
    first, last = tokens[0][0], tokens[-1][1]
    last_token = text[tokens[-1][0] : last]
    if _INITIAL.match(last_token.strip(",;:)")):
        return first, tokens[-1][0] + len(last_token.rstrip(",;:)"))
    core_end = first + len(text[first:last].rstrip(",;:)]}\"'”’|."))
    if text[first:core_end].endswith(("'s", "’s")):
        core_end -= 2
    return first, core_end


def name_tokens(value: str) -> list[str]:
    return [_core(t) for t in value.split() if _core(t)]


def _word_tokens(value: str) -> list[str]:
    """Name tokens that are words (not initials or particles)."""
    return [t for t in name_tokens(value) if not _INITIAL.match(t) and t.lower() not in PARTICLES]


def has_name_shape(value: str) -> bool:
    """2-4 tokens, each a capitalised word, an initial or a name particle; a vowel in
    every word; ends with a word. Shape only, no vocabulary."""
    tokens = name_tokens(value)
    if not 2 <= len(tokens) <= 4:
        return False
    for token in tokens:
        if _INITIAL.match(token) or token.lower() in PARTICLES:
            continue
        if not _NAME_WORD.match(token) or not re.search(r"[aeiouy]", token.lower()):
            return False
    return bool(_word_tokens(value)) and _NAME_WORD.match(tokens[-1]) is not None


def is_person_name(
    value: str,
    vocab: DocumentVocabulary,
    model_backed: bool = True,
    known_words: set[str] = frozenset(),
) -> bool:
    """A full name: name-shaped, no ordinary words (rule 1, unless confirmed by an
    email), and - if only our regex pattern found it - every word backed by evidence
    (rules 2-3)."""
    if not has_name_shape(value):
        return False
    words = _word_tokens(value)
    if any(vocab.is_common(w) for w in words):
        return False
    if model_backed:
        return True
    # Every word must be confirmed, so a pattern span cannot attach headings to a
    # real name ("Directors Bi-monthly Eleanor").
    return all(vocab.is_email_confirmed(w) or w in known_words for w in words)


_CAPITAL_WORD = re.compile(r"(?<![A-Za-z])[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*(?![A-Za-z])")
_GLUED_INITIAL = re.compile(r"[A-Z]\.$")
MAX_LEFT_TOKENS = 3


def surname_keys(full_names: Iterable[str], vocab: DocumentVocabulary) -> set[str]:
    """Name words that identify a person on their own: the last word of every accepted
    full name, plus name parts confirmed by the document's email addresses."""
    keys = {_letters(_word_tokens(name)[-1]) for name in full_names if _word_tokens(name)}
    keys |= vocab.email_names
    return {k for k in keys if len(k) >= 3 and k not in vocab.common_words}


def _token_before(text: str, start: int) -> tuple[int, str] | None:
    """The whitespace-separated token just before `start` on the same line."""
    gap = start
    while gap > 0 and text[gap - 1] in " 	":
        gap -= 1
    if gap == start or gap == 0:
        return None
    token_start = gap
    while token_start > 0 and not text[token_start - 1].isspace():
        token_start -= 1
    return token_start, text[token_start:gap]


def _extend_left(
    text: str, start: int, first_names: set[str], vocab: DocumentVocabulary | None
) -> int:
    """Grow a name to the left over its initials ("P. S.", glued "M.Achebe") and at most
    one first name: a first name of a person in this document, or a capitalised word
    that is not an ordinary word directly before an initial ("Ifeoma C. Adeyemi").
    Stops at anything else, so labels before a name ("Committee M. Achebe") stay out."""
    if start >= 2 and _GLUED_INITIAL.match(text[start - 2 : start]) and (
        start < 3 or not text[start - 3].isalpha()
    ):
        start -= 2
    took_initial = False
    for _ in range(MAX_LEFT_TOKENS):
        before = _token_before(text, start)
        if before is None:
            break
        token_start, token = before
        if _INITIAL.match(token):
            start, took_initial = token_start, True
            continue
        if _NAME_WORD.match(token) and (
            token in first_names
            or (took_initial and not (vocab and vocab.is_common(token)))
        ):
            start = token_start
        break
    return start


MIN_PAIR_COUNT = 2


def repeated_first_names(
    texts: Iterable[str], keys: set[str], vocab: DocumentVocabulary
) -> set[tuple[str, str]]:
    """(first name, surname) pairs that repeat in the document.

    A capitalised word that is not an ordinary word, written directly before a known
    surname and seen there at least twice, is a first name even when no email or full
    name confirms it ("Led by Priyanka Raghunathan"; the email is p.raghunathan@).
    One sighting is not enough: "Officer Wenzel" once is a label, not a name.
    """
    counts: Counter = Counter()
    for text in texts:
        for match in _CAPITAL_WORD.finditer(text or ""):
            surname = _letters(match.group())
            if surname not in keys:
                continue
            before = _token_before(text, match.start())
            if before is None:
                continue
            token = before[1]
            if (
                _NAME_WORD.match(token)
                and token.lower() not in EDGE_WORDS
                and _letters(token) not in keys
                and not vocab.is_common(token)
            ):
                counts[(token, surname)] += 1
    return {pair for pair, n in counts.items() if n >= MIN_PAIR_COUNT}


def surname_spans(
    text: str,
    keys: set[str],
    first_names: set[str] = frozenset(),
    vocab: DocumentVocabulary | None = None,
    first_name_pairs: set[tuple[str, str]] = frozenset(),
) -> list[tuple[int, int]]:
    """Every occurrence of a known surname, extended left to its initials / first name
    ("T. Lindqvist", "P. S. Raghavan", "Ifeoma C. Adeyemi", bare "Lindqvist")."""
    spans = []
    for match in _CAPITAL_WORD.finditer(text):
        surname = _letters(match.group())
        if surname not in keys:
            continue
        start = _extend_left(text, match.start(), first_names, vocab)
        if start == match.start() and first_name_pairs:
            before = _token_before(text, match.start())
            if before is not None and (before[1], surname) in first_name_pairs:
                start = before[0]
        spans.append((start, match.end()))
    return spans


def canonical_names(full_names: set[str]) -> set[str]:
    """Drop edge words that are not part of the name: if removing a name's first or last
    word leaves another full name found in the document ("Medium Helena Vorstermann"
    -> "Helena Vorstermann"), the extra word was a label next to it in one place."""
    result = set()
    for name in full_names:
        words = name.split()
        changed = True
        while changed and len(words) > 2:
            changed = False
            if " ".join(words[1:]) in full_names:
                words, changed = words[1:], True
            elif " ".join(words[:-1]) in full_names:
                words, changed = words[:-1], True
        result.add(" ".join(words))
    return result


def is_single_name_word(value: str, vocab: DocumentVocabulary) -> bool:
    """One capitalised word that is not an ordinary word in this document."""
    tokens = name_tokens(value)
    return (
        len(tokens) == 1
        and _NAME_WORD.match(tokens[0]) is not None
        and re.search(r"[aeiouy]", tokens[0].lower()) is not None
        and not vocab.is_common(tokens[0])
        and tokens[0].lower() not in EDGE_WORDS
    )
