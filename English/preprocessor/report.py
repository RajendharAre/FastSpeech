"""Drop accounting for preprocessing.

MODIFIED(both): NEW. Every clip that cannot be processed is assigned a
category (for example ``missing TextGrid``), written to
``preprocessing_log.csv`` and counted in ``preprocessing_summary.json``.
This module has no audio imports so it can be unit-tested directly.
"""
import json
from collections import Counter

LOG_COLUMNS = ("speaker", "clip", "split", "status", "category", "reason")


class ClipSkipped(Exception):
    """A manifest clip cannot be used; carries a category and a detail message."""

    def __init__(self, category, detail):
        self.category = category
        self.detail = detail
        super().__init__("{}: {}".format(category, detail))


def summarize(rows):
    """Return total/ok/dropped counts overall, per split and per category."""
    splits = ("train", "val")
    by_split = {
        split: {"total": 0, "ok": 0, "dropped": 0} for split in splits
    }
    by_category = Counter()

    for row in rows:
        split = row["split"]
        bucket = by_split.setdefault(split, {"total": 0, "ok": 0, "dropped": 0})
        bucket["total"] += 1
        if row["status"] == "ok":
            bucket["ok"] += 1
        else:
            bucket["dropped"] += 1
            by_category[row.get("category") or "unknown"] += 1

    dropped = sum(by_category.values())
    return {
        "total": len(rows),
        "ok": len(rows) - dropped,
        "dropped": dropped,
        "by_split": by_split,
        "by_category": dict(sorted(by_category.items())),
    }


def format_summary(summary):
    lines = [
        "Clips: {total} total, {ok} ok, {dropped} dropped".format(**summary)
    ]
    for split, counts in summary["by_split"].items():
        lines.append(
            "  {}: {} total, {} ok, {} dropped".format(
                split, counts["total"], counts["ok"], counts["dropped"]
            )
        )
    if summary["by_category"]:
        lines.append("Dropped by category:")
        for category, count in summary["by_category"].items():
            lines.append("  {}: {}".format(category, count))
    return "\n".join(lines)


def write_summary(path, summary):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
