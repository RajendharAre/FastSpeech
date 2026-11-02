"""Manifest loading for preprocessing.

Kept free of librosa/pyworld imports so it can be unit-tested without the
audio stack. Manifests are produced by ``scripts/split_dataset.py``.
"""
import csv
import math
import os

MANIFEST_COLUMNS = ("clip_id", "speaker", "duration", "transcript")


class ManifestError(ValueError):
    """Raised when a manifest is missing, empty or malformed."""


def has_path_separator(value):
    return "/" in value or "\\" in value or value in (".", "..")


def load_manifests(manifest_paths):
    """Return ``[(split, speaker, clip_id), ...]`` in manifest order.

    ``manifest_paths`` maps ``"train"``/``"val"`` to CSV paths. A clip may not
    appear twice, and may not appear in both manifests.
    """
    entries = []
    seen = {}

    for split in ("train", "val"):
        path = manifest_paths[split]
        if not os.path.exists(path):
            raise FileNotFoundError(
                "{} manifest not found: {}. Build split manifests with "
                "scripts/split_dataset.py first.".format(split, path)
            )

        count = 0
        with open(path, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ManifestError("{}: empty manifest".format(path))
            missing = [
                column for column in MANIFEST_COLUMNS if column not in reader.fieldnames
            ]
            if missing:
                raise ManifestError(
                    "{}: missing column(s): {}".format(path, ", ".join(missing))
                )

            for line_number, raw in enumerate(reader, start=2):
                values = {
                    column: (raw.get(column) or "").strip()
                    for column in MANIFEST_COLUMNS
                }
                if not any(values.values()):
                    continue

                for column in ("clip_id", "speaker"):
                    if not values[column]:
                        raise ManifestError(
                            "{}:{}: column '{}' is empty".format(
                                path, line_number, column
                            )
                        )
                    if has_path_separator(values[column]):
                        raise ManifestError(
                            "{}:{}: '{}' must not be a path: {!r}".format(
                                path, line_number, column, values[column]
                            )
                        )

                try:
                    duration = float(values["duration"])
                except ValueError:
                    raise ManifestError(
                        "{}:{}: duration {!r} is not a number".format(
                            path, line_number, values["duration"]
                        )
                    )
                if not math.isfinite(duration) or duration <= 0:
                    raise ManifestError(
                        "{}:{}: duration must be finite and > 0".format(
                            path, line_number
                        )
                    )
                if not values["transcript"]:
                    raise ManifestError(
                        "{}:{}: transcript is empty".format(path, line_number)
                    )

                key = (values["speaker"], values["clip_id"])
                if key in seen:
                    raise ManifestError(
                        "{}:{}: duplicate clip {}/{} (already in the {} "
                        "manifest)".format(
                            path,
                            line_number,
                            key[0],
                            key[1],
                            seen[key],
                        )
                    )
                seen[key] = split
                entries.append((split, key[0], key[1]))
                count += 1

        if count == 0:
            raise ManifestError("{}: manifest contains no rows".format(path))

    return entries
