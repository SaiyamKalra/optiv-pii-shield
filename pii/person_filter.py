import re
from collections import Counter
from collections.abc import Iterable

from config.settings import CORP_EMAIL_PATTERN

EDGE_WORDS = {"by", "of", "the", "and", "to", "at", "for", "with", "via", "from", "on", "in"}

PARTICLES = {"van", "von", "der", "den", "de", "del", "da", "di", "du", "la", "le", "bin", "ibn", "al"}

_TOKEN = re.compile(r"\S+")
_INITIAL = re.compile(r"^[A-Z]\.(?:-?[A-Z]\.)?$")
_NAME_WORD = re.compile(r"^[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*$")
_EMAIL = re.compile(CORP_EMAIL_PATTERN.regex)
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_LOWER_WORD = re.compile(r"(?<![A-Za-z])[a-z][a-z'’-]*[a-z](?![A-Za-z])")
_ANY_WORD = re.compile(r"(?<![A-Za-z])[A-Za-z][A-Za-z'’-]*[A-Za-z](?![A-Za-z])")

COMMON_LOWERCASE_MIN = 2
COMMON_LOWERCASE_SHARE = 0.25


def _keys(word: str) -> list[str]:
    word = word.lower().replace("’", "'").removesuffix("'s")
    return [word] + [part for part in word.split("-") if len(part) > 1 and part != word]


def _letters(token: str) -> str:
    return re.sub(r"[^a-z]", "", token.lower())


class DocumentVocabulary:
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
        self.common_words: set[str] = {
            word for word, n in lower_counts.items()
            if n >= COMMON_LOWERCASE_MIN or n / max(n, all_counts[word]) >= COMMON_LOWERCASE_SHARE
        }
        for local in locals_:
            pieces = [p for p in re.split(r"[._\-+\d]+", local.lower()) if p]
            parts = [p for p in pieces if len(p) >= 3]
            if len(pieces) == 2 and len(pieces[0]) == 1 and parts:
                self.email_names.update(parts)
            else:
                self.email_names.update(p for p in parts if p not in self.common_words)

    def is_common(self, token: str) -> bool:
        lower = token.lower().replace("’", "'").removesuffix("'s")
        return lower in self.common_words and not self.is_email_confirmed(token)

    def is_email_confirmed(self, token: str) -> bool:
        letters = _letters(token)
        return len(letters) >= 3 and letters in self.email_names


def _core(token: str) -> str:
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
    if stripped.endswith(".") and not _INITIAL.match(stripped) and len(core) <= 4:
        return True
    return core.isupper() and len(core) >= 2 and not _INITIAL.match(core)


def trim_person_span(
    text: str, start: int, end: int, vocab: DocumentVocabulary
) -> tuple[int, int]:
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
    return [t for t in name_tokens(value) if not _INITIAL.match(t) and t.lower() not in PARTICLES]


def has_name_shape(value: str) -> bool:
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
    if not has_name_shape(value):
        return False
    words = _word_tokens(value)
    if any(vocab.is_common(w) for w in words):
        return False
    if model_backed:
        return True
    return all(vocab.is_email_confirmed(w) or w in known_words for w in words)


_CAPITAL_WORD = re.compile(r"(?<![A-Za-z])[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*(?![A-Za-z])")
_GLUED_INITIAL = re.compile(r"[A-Z]\.$")
MAX_LEFT_TOKENS = 3


def surname_keys(full_names: Iterable[str], vocab: DocumentVocabulary) -> set[str]:
    keys = {_letters(_word_tokens(name)[-1]) for name in full_names if _word_tokens(name)}
    keys |= vocab.email_names
    return {k for k in keys if len(k) >= 3 and k not in vocab.common_words}


def _token_before(text: str, start: int) -> tuple[int, str] | None:
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
    tokens = name_tokens(value)
    return (
        len(tokens) == 1
        and _NAME_WORD.match(tokens[0]) is not None
        and re.search(r"[aeiouy]", tokens[0].lower()) is not None
        and not vocab.is_common(tokens[0])
        and tokens[0].lower() not in EDGE_WORDS
    )
