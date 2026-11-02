""" from https://github.com/keithito/tacotron (MIT), adapted for Telugu.

Unlike the reference English frontend, Telugu is encoded character by
character over the frozen Unicode set in ``text.symbols``. Curly braces mark
alignment input produced from a TextGrid, not ARPAbet phonemes.
"""
from text import cleaners
from text import telugu
from text.symbols import symbols

# Kept for callers that inspect the Telugu symbol table.
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names):
    """Convert text to a list of symbol IDs.

    Plain text is encoded one character at a time. Text wrapped in curly
    braces is treated as alignment tokens (one character per phone).
    """
    text = _clean_text(text, cleaner_names)
    if "{" in text or "}" in text:
        return telugu.encode_alignment_tokens(text)
    return telugu.encode_text(text)


def sequence_to_text(sequence):
    """Convert a sequence of symbol IDs back to text."""
    return telugu.decode_text(sequence)


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        if not hasattr(cleaners, name):
            raise ValueError("Unknown cleaner: {}".format(name))
        text = getattr(cleaners, name)(text)
    return text
