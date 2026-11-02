"""Telugu text frontend (character level).

The reference implementation (ming024/FastSpeech2, MIT) ships an English
frontend. Telugu is encoded over a frozen Unicode symbol set
(``text.symbols``) so the same text always produces the same ID sequence.

Normalization policy:

* NFC composes canonical equivalents, e.g. U+0C46 U+0C56 -> U+0C48, so a
  character has exactly one representation.
* All Unicode whitespace collapses to a single ASCII space.
* Zero-width and format controls (ZWJ, ZWNJ, BOM, variation selectors) are
  removed; they carry no phonemic information for this model.
* Common typographic punctuation is folded onto ASCII punctuation that the
  symbol set supports.
* Nothing is transliterated and no unsupported character is silently dropped:
  encoding raises ``UnknownCharacterError`` instead.

Alignment input (text produced by the preprocessor from a TextGrid) is written
as ``{phone phone ...}``. Because the frontend is character level, every
non-silence phone label must normalize to exactly one character; otherwise the
text sequence and the duration sequence would have different lengths.
"""
import re
import unicodedata

from text.symbols import symbols

_SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(symbols)}
_ID_TO_SYMBOL = {index: symbol for index, symbol in enumerate(symbols)}

if len(_SYMBOL_TO_ID) != len(symbols):
    raise RuntimeError("the Telugu symbol set contains duplicates")

SILENCE_TOKENS = ("sp", "spn", "sil")
SILENCE_SYMBOLS = {"sp": "@sp", "spn": "@spn", "sil": "@sil"}
_SILENCE_TEXT = {"@sp": "sp", "@spn": "spn", "@sil": "sil"}

_PUNCTUATION_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2033": '"',
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    "\u0964": ".",
    "\u0965": ".",
    "\u2026": "...",
}
# Format/control characters carry no Telugu phonemic information. Private-use,
# unassigned and symbol characters are deliberately not dropped: they reach
# encode_text and raise UnknownCharacterError.
_FORMAT_CATEGORIES = frozenset({"Cf", "Cc", "Cs"})


class UnknownCharacterError(ValueError):
    """Raised when text contains a character outside the Telugu symbol set."""


class AlignmentTokenError(ValueError):
    """Raised when a phone label cannot be mapped one-to-one to a symbol."""


class UnknownSymbolError(ValueError):
    """Raised when a sequence contains an ID outside the Telugu symbol set."""


def _is_variation_selector(character):
    code = ord(character)
    return 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF


def _describe_unknown(characters):
    details = []
    for character in characters:
        details.append(
            "{!r} (U+{:04X}, {})".format(
                character,
                ord(character),
                unicodedata.name(character, "unnamed"),
            )
        )
    return "Telugu frontend does not support: {}".format(", ".join(details))


def normalize_text(text):
    """Return the canonical form used by training and inference."""
    if not isinstance(text, str):
        raise TypeError("text must be str, got {}".format(type(text).__name__))

    text = unicodedata.normalize("NFC", text)
    pieces = []
    for character in text:
        if character in _PUNCTUATION_MAP:
            pieces.append(_PUNCTUATION_MAP[character])
        elif character.isspace():
            pieces.append(" ")
        elif _is_variation_selector(character):
            continue
        elif unicodedata.category(character) in _FORMAT_CATEGORIES:
            continue
        else:
            pieces.append(character)

    return re.sub(r" +", " ", "".join(pieces)).strip()


def find_unknown_characters(text):
    """Return sorted characters that ``encode_text`` would reject."""
    normalized = normalize_text(text)
    return sorted({character for character in normalized if character not in _SYMBOL_TO_ID})


def encode_text(text):
    """Encode plain Telugu text to a list of symbol IDs."""
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("text is empty after normalization")

    unknown = find_unknown_characters(normalized)
    if unknown:
        raise UnknownCharacterError(_describe_unknown(unknown))
    return [_SYMBOL_TO_ID[character] for character in normalized]


def decode_text(sequence):
    """Decode symbol IDs back to text. Padding IDs are ignored."""
    pieces = []
    for symbol_id in sequence:
        try:
            symbol = _ID_TO_SYMBOL[symbol_id]
        except (KeyError, TypeError):
            raise UnknownSymbolError(
                "symbol id {!r} is not in the Telugu symbol set".format(symbol_id)
            )

        if symbol == "_":
            continue
        if symbol in _SILENCE_TEXT:
            pieces.append(" {} ".format(_SILENCE_TEXT[symbol]))
        else:
            pieces.append(symbol)

    return re.sub(r" +", " ", "".join(pieces)).strip()


def encode_alignment_tokens(text):
    """Encode ``{phone phone ...}`` input produced from a TextGrid."""
    normalized = normalize_text(text)
    if not (normalized.startswith("{") and normalized.endswith("}")):
        raise AlignmentTokenError(
            "alignment must be wrapped in one pair of braces, got {!r}".format(normalized)
        )

    inner = normalized[1:-1]
    if "{" in inner or "}" in inner:
        raise AlignmentTokenError(
            "alignment contains nested or unbalanced braces: {!r}".format(normalized)
        )

    tokens = inner.split()
    if not tokens:
        raise AlignmentTokenError("alignment is empty: {!r}".format(normalized))

    sequence = []
    for token in tokens:
        name = token[1:] if token.startswith("@") else token
        if name.lower() in SILENCE_TOKENS:
            sequence.append(_SYMBOL_TO_ID[SILENCE_SYMBOLS[name.lower()]])
            continue

        if len(token) != 1:
            raise AlignmentTokenError(
                "alignment token {!r} normalizes to {} characters; the Telugu "
                "frontend is character level and requires one character per "
                "phone so text and duration lengths match".format(
                    token, len(token)
                )
            )
        sequence.extend(encode_text(token))

    return sequence
