"""Rule-based English text normalisation.

Returns the normalised text together with the names of the rules that changed
it. Case is preserved so acronyms can be matched; the frontend cleaners
lowercase the result afterwards.
"""
import csv
import os
import re

from text.numbers import normalize_numbers

DEFAULT_ACRONYMS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "acronyms.csv"
)

_CURRENCY_UNITS = {
    "£": ("pound", "pounds"),
    "$": ("dollar", "dollars"),
    "€": ("euro", "euros"),
    "₹": ("rupee", "rupees"),
}
_DIGIT_WORDS = (
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
)
_MONTHS = (
    "", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_ABBREVIATIONS = {
    "e.g.": "for example",
    "i.e.": "that is",
    "etc.": "et cetera",
    "vs.": "versus",
    "Drs.": "doctors",
    "Mrs.": "missus",
    "Mr.": "mister",
    "Ms.": "miss",
    "Dr.": "doctor",
    "Prof.": "professor",
    "St.": "saint",
    "Jr.": "junior",
    "Sr.": "senior",
    "Ltd.": "limited",
    "Co.": "company",
    "Corp.": "corporation",
    "Inc.": "incorporated",
    "Gov.": "governor",
    "Sen.": "senator",
    "Rep.": "representative",
    "Gen.": "general",
    "Col.": "colonel",
    "Capt.": "captain",
    "Lt.": "lieutenant",
    "Sgt.": "sergeant",
    "Rev.": "reverend",
    "Hon.": "honorable",
    "Esq.": "esquire",
    "Ft.": "fort",
    "Maj.": "major",
}
_ABBREVIATION_RE = re.compile(
    r"\b(?:{})".format(
        "|".join(
            re.escape(key)
            for key in sorted(_ABBREVIATIONS, key=len, reverse=True)
        )
    ),
    re.IGNORECASE,
)
_DOTTED_ACRONYM_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}")
_CURRENCY_RE = re.compile(r"([£$€₹])\s*(\d[\d,]*(?:\.\d+)?)")
_PERCENT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*%")
_TIME_RE = re.compile(
    r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)(?!:\d{2})"
    r"(?:\s*([AaPp])\.?\s*([Mm])(?!\w))?"
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DECIMAL_RE = re.compile(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])")
_WHITESPACE_RE = re.compile(r"\s+")


def load_acronyms(path=DEFAULT_ACRONYMS_PATH):
    """Read ``acronyms.csv`` into ``{acronym: spoken}``."""
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        required = {"acronym", "spoken", "mode"}
        missing = required - columns
        if missing:
            raise ValueError(
                "{}: missing column(s): {}".format(path, ", ".join(sorted(missing)))
            )

        mapping = {}
        for line_number, row in enumerate(reader, start=2):
            acronym = (row.get("acronym") or "").strip()
            spoken = (row.get("spoken") or "").strip()
            mode = (row.get("mode") or "").strip().lower()
            if not acronym or not spoken:
                raise ValueError(
                    "{}:{}: acronym and spoken must be non-empty".format(
                        path, line_number
                    )
                )
            if mode not in {"letters", "word"}:
                raise ValueError(
                    "{}:{}: mode must be 'letters' or 'word', got {!r}".format(
                        path, line_number, mode
                    )
                )
            if acronym in mapping:
                raise ValueError(
                    "{}:{}: duplicate acronym {!r}".format(path, line_number, acronym)
                )
            mapping[acronym] = spoken
    return mapping


def _digits_to_words(digits):
    return " ".join(_DIGIT_WORDS[int(digit)] for digit in digits)


def _amount_to_words(amount):
    whole, _, fraction = amount.replace(",", "").partition(".")
    words = normalize_numbers(whole)
    if fraction:
        words += " point " + _digits_to_words(fraction)
    return words


def _ordinal_words(number):
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return normalize_numbers("{}{}".format(number, suffix))


def _expand_currency(match):
    symbol, amount = match.group(1), match.group(2)
    singular, plural = _CURRENCY_UNITS[symbol]
    whole = amount.replace(",", "").split(".")[0]
    unit = singular if int(whole) == 1 else plural
    return "{} {}".format(_amount_to_words(amount), unit)


def _expand_percent(match):
    return "{} percent".format(_amount_to_words(match.group(1)))


def _expand_time(match):
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return match.group(0)

    words = normalize_numbers(str(hour))
    if minute == 0:
        words += " o'clock"
    elif minute < 10:
        words += " oh " + normalize_numbers(str(minute))
    else:
        words += " " + normalize_numbers(str(minute))

    if match.group(3):
        words += " " + match.group(3).lower() + " " + match.group(4).lower()
    return words


def _expand_iso_date(match):
    year, month, day = (int(part) for part in match.groups())
    if not (1000 <= year <= 2999 and 1 <= month <= 12 and 1 <= day <= 31):
        return match.group(0)
    return "the {} of {} {}".format(
        _ordinal_words(day), _MONTHS[month], normalize_numbers(str(year))
    )


def _expand_abbreviation(match):
    key = match.group(0)
    for source, replacement in _ABBREVIATIONS.items():
        if source.lower() == key.lower():
            return replacement
    return key


def _expand_decimal(match):
    return "{} point {}".format(
        normalize_numbers(match.group(1)), _digits_to_words(match.group(2))
    )


def normalise_english(text, acronyms_path=DEFAULT_ACRONYMS_PATH):
    """Return ``(normalised_text, fired_rules)`` for English text."""
    acronyms = load_acronyms(acronyms_path)
    acronym_re = None
    if acronyms:
        acronym_re = re.compile(
            r"\b(?:{})\b".format(
                "|".join(
                    re.escape(key)
                    for key in sorted(acronyms, key=len, reverse=True)
                )
            )
        )

    rules = []

    def apply(rule, transform):
        nonlocal text
        updated = transform(text)
        if updated != text:
            rules.append(rule)
            text = updated

    apply("currency", lambda value: _CURRENCY_RE.sub(_expand_currency, value))
    apply("percent", lambda value: _PERCENT_RE.sub(_expand_percent, value))
    apply("time", lambda value: _TIME_RE.sub(_expand_time, value))
    apply("date", lambda value: _ISO_DATE_RE.sub(_expand_iso_date, value))
    apply(
        "abbreviation",
        lambda value: _ABBREVIATION_RE.sub(_expand_abbreviation, value),
    )

    def expand_acronyms(value):
        if acronym_re is not None:
            value = acronym_re.sub(lambda match: acronyms[match.group(0)], value)
        return _DOTTED_ACRONYM_RE.sub(
            lambda match: " ".join(
                character for character in match.group(0) if character.isalpha()
            ),
            value,
        )

    apply("acronym", expand_acronyms)

    def expand_symbols(value):
        value = re.sub(r"\s*°C\b", " degrees celsius", value)
        value = re.sub(r"\s*°F\b", " degrees fahrenheit", value)
        value = re.sub(r"\s*&\s*", " and ", value)
        value = re.sub(r"\s*\+\s*", " plus ", value)
        value = re.sub(r"\s*=\s*", " equals ", value)
        value = re.sub(r"\s*@\s*", " at ", value)
        value = re.sub(r"#(?=\s*\d)", "number ", value)
        value = value.replace("×", " times ").replace("÷", " divided by ")
        value = value.replace("°", " degrees ")
        return value

    apply("symbol", expand_symbols)
    apply("decimal", lambda value: _DECIMAL_RE.sub(_expand_decimal, value))

    def expand_numbers(value):
        if not re.search(r"\d", value):
            return value
        return normalize_numbers(value)

    apply("number", expand_numbers)
    apply("whitespace", lambda value: _WHITESPACE_RE.sub(" ", value).strip())
    return text, tuple(rules)


def format_rules(rules):
    """Format fired rule names for a manifest cell."""
    return ";".join(rules)
