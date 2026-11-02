""" from https://github.com/keithito/tacotron (MIT), adapted for Telugu.

The reference implementation defines an English symbol set (ASCII letters,
ARPAbet and pinyin). Telugu cannot be transliterated to that set without
losing information, so this module defines a character-level Telugu set.

The Telugu code points are listed as explicit ranges rather than generated
from ``unicodedata``: symbol IDs are baked into checkpoints, so the set must
not change when the Python/Unicode version changes. The ranges below cover
every assigned letter, combining mark and digit in the Telugu block
(U+0C00-U+0C7F); unassigned code points and the two Telugu punctuation
characters (U+0C77, U+0C7F) are excluded.
"""

_pad = "_"
_special = "-"
_punctuation = "!'(),.:;? \""
_ascii_digits = "0123456789"
_silences = ["@sp", "@spn", "@sil"]

# Assigned Telugu letters (Lo), signs (Mn/Mc) and digits (Nd/No).
_TELUGU_RANGES = (
    (0x0C00, 0x0C04),
    (0x0C05, 0x0C0C),
    (0x0C0E, 0x0C10),
    (0x0C12, 0x0C28),
    (0x0C2A, 0x0C39),
    (0x0C3C, 0x0C44),
    (0x0C46, 0x0C48),
    (0x0C4A, 0x0C4D),
    (0x0C55, 0x0C56),
    (0x0C58, 0x0C5A),
    (0x0C5D, 0x0C5D),
    (0x0C60, 0x0C63),
    (0x0C66, 0x0C6F),
    (0x0C78, 0x0C7E),
)
_telugu = [chr(code) for low, high in _TELUGU_RANGES for code in range(low, high + 1)]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_ascii_digits)
    + _telugu
    + _silences
)

assert len(symbols) == len(set(symbols)), "duplicate symbol in the Telugu set"
