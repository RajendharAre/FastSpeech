"""Pure helpers for kaggle/inspect_datasets.py.

No Hugging Face imports here, so these functions can be unit-tested on a laptop
with synthetic data (see kaggle/tests/test_inspect_utils.py).
"""
import io
import re
import statistics
import unicodedata
import wave

import numpy as np

# Buckets are reported as counts of clips, not as claims about the data.
DURATION_BUCKETS = [
    (0.0, 1.0, "< 1 s"),
    (1.0, 3.0, "1-3 s"),
    (3.0, 10.0, "3-10 s"),
    (10.0, 30.0, "10-30 s"),
    (30.0, float("inf"), ">= 30 s"),
]

TAG_PATTERNS = {
    "angle_brackets": re.compile(r"<[^<>]{1,40}>"),
    "square_brackets": re.compile(r"\[[^\[\]]{1,40}\]"),
    "curly_braces": re.compile(r"\{[^{}]{1,40}\}"),
    "parens": re.compile(r"\([^()]{1,40}\)"),
    "percent": re.compile(r"%"),
    "ellipsis": re.compile(r"\.\.\."),
}


def probe_audio_bytes(raw):
    """Read duration/sampling rate from encoded audio bytes without decoding.

    Returns {"frames", "samplerate", "format"} or {"error": "..."}.
    soundfile handles WAV/FLAC/OGG/MP3; the stdlib wave module is a WAV fallback.
    """
    if not raw:
        return {"error": "no bytes"}
    try:
        import soundfile as sf

        with sf.SoundFile(io.BytesIO(raw)) as f:
            return {
                "frames": int(f.frames),
                "samplerate": int(f.samplerate),
                "format": str(f.format),
            }
    except Exception as sf_error:
        try:
            with wave.open(io.BytesIO(raw)) as w:
                return {
                    "frames": int(w.getnframes()),
                    "samplerate": int(w.getframerate()),
                    "format": "WAV",
                }
        except Exception as wave_error:
            return {
                "error": "soundfile: {}; wave: {}".format(sf_error, wave_error)
            }


def probe_audio_value(value):
    """Handle the shapes the `datasets` Audio feature can return.

    - decode=False: {"path": ..., "bytes": ...}
    - decode=True:  {"path": ..., "array": ..., "sampling_rate": ...}
    - plain bytes / str path
    """
    if value is None:
        return {"error": "no value"}
    if isinstance(value, (bytes, bytearray)):
        return probe_audio_bytes(bytes(value))
    if isinstance(value, str):
        try:
            with open(value, "rb") as f:
                return probe_audio_bytes(f.read())
        except OSError as e:
            return {"error": "cannot read {}: {}".format(value, e)}
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return probe_audio_bytes(value["bytes"])
        if value.get("path"):
            try:
                with open(value["path"], "rb") as f:
                    return probe_audio_bytes(f.read())
            except OSError as e:
                return {"error": "cannot read {}: {}".format(value["path"], e)}
        if value.get("array") is not None:
            array = value["array"]
            rate = value.get("sampling_rate")
            if rate:
                return {
                    "frames": int(len(array)),
                    "samplerate": int(rate),
                    "format": "decoded array",
                }
    return {"error": "unrecognised audio value: {}".format(type(value).__name__)}


def duration_stats(durations):
    """Summary of clip durations in seconds."""
    if not durations:
        return {"n": 0}
    values = np.asarray(durations, dtype=float)
    stats = {
        "n": int(values.size),
        "total_seconds": float(values.sum()),
        "total_hours": float(values.sum() / 3600.0),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(np.median(values)),
    }
    for p in (1, 5, 25, 75, 95, 99):
        stats["p{}".format(p)] = float(np.percentile(values, p))
    stats["buckets"] = [
        {
            "label": label,
            "count": int(((values >= low) & (values < high)).sum()),
        }
        for low, high, label in DURATION_BUCKETS
    ]
    return stats


def text_stats(values, sample_count=3, seed=1234):
    """Character inventory, duplication and tag counts for one text column."""
    strings = [v for v in values if isinstance(v, str)]
    lengths = [len(s) for s in strings]
    distinct_chars = {}
    non_ascii = 0
    total_chars = 0
    tag_counts = {name: 0 for name in TAG_PATTERNS}

    for s in strings:
        total_chars += len(s)
        for ch in s:
            distinct_chars[ch] = distinct_chars.get(ch, 0) + 1
            if ord(ch) > 127:
                non_ascii += 1
        for name, pattern in TAG_PATTERNS.items():
            tag_counts[name] += len(pattern.findall(s))

    top_chars = sorted(distinct_chars.items(), key=lambda kv: -kv[1])[:30]
    top_chars_out = []
    for ch, count in top_chars:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = "UNNAMED"
        top_chars_out.append(
            {
                "char": ch,
                "codepoint": "U+{:04X}".format(ord(ch)),
                "name": name,
                "count": int(count),
            }
        )

    counts = {}
    for s in strings:
        counts[s] = counts.get(s, 0) + 1
    duplicate_values = [s for s, c in counts.items() if c > 1]
    duplicate_rows = sum(c for c in counts.values() if c > 1)

    rng = np.random.default_rng(seed)
    non_empty = [s for s in strings if s.strip()]
    if len(non_empty) > sample_count:
        idx = rng.choice(len(non_empty), size=sample_count, replace=False)
        samples = [non_empty[i] for i in idx]
    else:
        samples = non_empty[:sample_count]

    return {
        "n": len(strings),
        "empty": int(sum(1 for s in strings if not s.strip())),
        "length_mean": float(statistics.mean(lengths)) if lengths else 0.0,
        "length_median": float(statistics.median(lengths)) if lengths else 0.0,
        "length_max": int(max(lengths)) if lengths else 0,
        "distinct_chars": int(len(distinct_chars)),
        "non_ascii_share": float(non_ascii / total_chars) if total_chars else 0.0,
        "top_chars": top_chars_out,
        "tag_counts": tag_counts,
        "distinct_values": int(len(counts)),
        "duplicate_values": int(len(duplicate_values)),
        "duplicate_rows": int(duplicate_rows),
        "samples": samples,
    }


def group_stats(values, durations):
    """Clip count and total hours for every value of a candidate group column.

    A duration of None means its audio header could not be read; those clips are
    counted but contribute no hours, and `missing` records the gap.
    """
    groups = {}
    for value, duration in zip(values, durations):
        key = "<null>" if value is None else str(value)
        entry = groups.setdefault(
            key, {"value": key, "clips": 0, "seconds": 0.0, "missing": 0}
        )
        entry["clips"] += 1
        if duration is None:
            entry["missing"] += 1
        else:
            entry["seconds"] += float(duration)
    out = list(groups.values())
    for entry in out:
        entry["hours"] = entry["seconds"] / 3600.0
    out.sort(key=lambda e: -e["seconds"])
    return out


def duplicate_across_groups(values, groups):
    """How many distinct text values occur under more than one group value."""
    seen = {}
    for text, group in zip(values, groups):
        key = "<null>" if group is None else str(group)
        seen.setdefault(text, set()).add(key)
    return sum(1 for g in seen.values() if len(g) > 1)


def md_escape(value):
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def fmt_num(value, digits=3):
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return ("{:,." + str(digits) + "f}").format(value)
    return "{:,}".format(value)


def fmt_hours(seconds):
    return "not measured" if seconds is None else "{:.4f} h".format(seconds / 3600.0)


def render_markdown(reports, generated_at):
    """Render the combined dataset report from the JSON-able report dicts."""
    lines = [
        "# Dataset report",
        "",
        "Generated by `kaggle/inspect_datasets.py` on **{}** (UTC).".format(generated_at),
        "Commands: "
        + "; ".join("`{}`".format(r.get("command", "?")) for r in reports),
        "",
        "Every number below was written by that script from the data it loaded.",
        "Where a measurement failed, the report says **not measured** and gives the",
        "reason. Nothing here is taken from a dataset card, except the licence and",
        "gating fields, which are quoted from the Hugging Face API and labelled as such.",
        "",
        "Sources: " + ", ".join(
            "`{}`".format(r.get("json_path", "?")) for r in reports
        ),
        "",
    ]

    for report in reports:
        lines += render_dataset(report)
        lines += ["", "---", ""]
    return "\n".join(lines).rstrip() + "\n"


def render_dataset(r):
    m = r["meta"]
    s = r["summary"]
    lines = [
        "## {} ({})".format(m["repo_id"], m["language"]),
        "",
        "### Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        "| Source | https://huggingface.co/datasets/{} |".format(m["repo_id"]),
        "| Revision (sha) | `{}` |".format(md_escape(m.get("sha"))),
        "| Last modified | {} |".format(md_escape(m.get("last_modified"))),
        "| Inspected at | {} |".format(r["generated_at"]),
        "| Python | {} |".format(md_escape(m.get("python_version"))),
        "| datasets | {} |".format(md_escape(m.get("datasets_version"))),
        "| huggingface_hub | {} |".format(md_escape(m.get("hub_version"))),
        "| Rows inspected | {} |".format(fmt_num(s.get("rows"))),
        "",
        "### Licence and gating (from the Hugging Face API, not parsed by hand)",
        "",
        "| Field | Value |",
        "|---|---|",
        "| `gated` | `{}` |".format(md_escape(m.get("gated"))),
        "| `license` (card) | `{}` |".format(md_escape(m.get("license"))),
        "| licence/language tags | {} |".format(
            md_escape(", ".join(m.get("tags") or []) or "none returned")
        ),
        "| Card | https://huggingface.co/datasets/{} |".format(m["repo_id"]),
        "",
    ]
    if m.get("card_error"):
        lines += [
            "> Card metadata could not be read: {}".format(md_escape(m["card_error"])),
            "",
        ]
    if m.get("licence_note"):
        lines += ["> {}".format(m["licence_note"]), ""]

    lines += [
        "### Structure (enumerated, not assumed)",
        "",
        "| Config | Split | Rows | Features |",
        "|---|---|---|---|",
    ]
    for split in r["splits"]:
        lines.append(
            "| {} | {} | {} | {} |".format(
                md_escape(split["config"]),
                md_escape(split["split"]),
                fmt_num(split["num_rows"]),
                md_escape(split["features"]),
            )
        )
    lines += [""]

    if r.get("label_columns"):
        lines += [
            "ClassLabel columns: "
            + ", ".join(
                "`{}` = {}".format(k, ", ".join(str(x) for x in v))
                for k, v in r["label_columns"].items()
            ),
            "",
        ]
    if r.get("summary", {}).get("group_column"):
        lines += [
            "Cross-group duplicate check uses column `{}` (best speaker-like "
            "low-cardinality column found).".format(
                md_escape(r["summary"]["group_column"])
            ),
            "",
        ]

    lines += ["### Audio", ""]
    if not r["audio"]["columns"]:
        lines += ["No `Audio` feature column was found. Nothing measured.", ""]
    else:
        lines += [
            "| Audio column | Clips measured | Failed | Sampling rates (Hz) | Total hours |",
            "|---|---|---|---|---|",
        ]
        for col in r["audio"]["columns"]:
            rates = ", ".join(
                "{} ({})".format(rate, count)
                for rate, count in sorted(
                    col["sampling_rates"].items(), key=lambda kv: int(kv[0])
                )
            )
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    md_escape(col["name"]),
                    fmt_num(col["measured"]),
                    fmt_num(col["failed"]),
                    md_escape(rates or "not measured"),
                    fmt_num(col.get("total_hours"), 4)
                    if col.get("total_hours") is not None
                    else "not measured",
                )
            )
        if r.get("summary", {}).get("total_hours") is not None:
            lines += [
                "",
                "Measured total: **{:.4f} h** across {} clips (primary audio column "
                "`{}`).".format(
                    r["summary"]["total_hours"],
                    fmt_num(r["summary"].get("clips_measured")),
                    md_escape(r["audio"].get("primary_column")),
                ),
            ]

        dist = r["audio"].get("duration_stats")
        if dist and dist.get("n"):
            lines += [
                "",
                "**Duration distribution (seconds, measured from audio headers)**",
                "",
                "| n | min | p5 | p25 | median | p75 | p95 | max | mean | std |",
                "|---|---|---|---|---|---|---|---|---|---|",
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    fmt_num(dist["n"]),
                    fmt_num(dist["min"]),
                    fmt_num(dist["p5"]),
                    fmt_num(dist["p25"]),
                    fmt_num(dist["median"]),
                    fmt_num(dist["p75"]),
                    fmt_num(dist["p95"]),
                    fmt_num(dist["max"]),
                    fmt_num(dist["mean"]),
                    fmt_num(dist["std"]),
                ),
                "",
                "| Bucket | Clips |",
                "|---|---|",
            ]
            for bucket in dist["buckets"]:
                lines.append(
                    "| {} | {} |".format(bucket["label"], fmt_num(bucket["count"]))
                )
        else:
            lines += ["", "Duration distribution: **not measured**."]

        if r["audio"].get("metadata_duration_columns"):
            lines += ["", "Numeric columns whose name contains 'dur' (units not assumed):", ""]
            lines += [
                "- `{}`: n={}, min={}, max={}, mean={}".format(
                    c["name"], c["n"], fmt_num(c["min"]), fmt_num(c["max"]), fmt_num(c["mean"])
                )
                for c in r["audio"]["metadata_duration_columns"]
            ]
        lines += [""]

    lines += ["### Group breakdown (for every low-cardinality column)", ""]
    if not r["groups"]:
        lines += ["No column had a low enough cardinality to group. Not measured.", ""]
    for group in r["groups"]:
        lines += [
            "**Column `{}`** ({} distinct values)".format(
                md_escape(group["column"]), fmt_num(group["distinct"])
            ),
            "",
            "| Value | Clips | Hours |",
            "|---|---|---|",
        ]
        for entry in group["top"]:
            lines.append(
                "| {} | {} | {:.4f} |".format(
                    md_escape(entry["value"]), fmt_num(entry["clips"]), entry["hours"]
                )
            )
        if group["distinct"] > len(group["top"]):
            lines.append(
                "| _showing {} of {}_ | | |".format(
                    len(group["top"]), fmt_num(group["distinct"])
                )
            )
        if group.get("csv_path"):
            lines += ["", "Full table: `{}`".format(group["csv_path"])]
        if group.get("clips_without_duration"):
            lines += [
                "",
                "> {} clip(s) in this table have no measured duration, so their "
                "hours are missing from the totals.".format(
                    fmt_num(group["clips_without_duration"])
                ),
            ]
        lines += [""]

    lines += ["### Text columns", ""]
    if not r["text"]:
        lines += ["No string column was found. Not measured.", ""]
    for text in r["text"]:
        lines += [
            "**Column `{}`**".format(md_escape(text["column"])),
            "",
            "| n | empty | distinct values | rows in duplicate values | "
            "mean length | median length | max length | distinct characters | non-ASCII share |",
            "|---|---|---|---|---|---|---|---|---|",
            "| {} | {} | {} | {} | {} | {} | {} | {} | {:.4f} |".format(
                fmt_num(text["n"]),
                fmt_num(text["empty"]),
                fmt_num(text["distinct_values"]),
                fmt_num(text["duplicate_rows"]),
                fmt_num(text["length_mean"], 2),
                fmt_num(text["length_median"], 2),
                fmt_num(text["length_max"]),
                fmt_num(text["distinct_chars"]),
                text["non_ascii_share"],
            ),
            "",
            "Tag-like pattern counts: "
            + ", ".join(
                "`{}`={}".format(k, fmt_num(v))
                for k, v in text["tag_counts"].items()
            ),
            "",
            "Top characters: "
            + ", ".join(
                "`{}` ({} {}) x{}".format(
                    c["char"], c["codepoint"], c["name"], fmt_num(c["count"])
                )
                for c in text["top_chars"][:15]
            ),
            "",
            "Duplicate transcripts across groups (`{}`): {}. "
            "Total duplicate transcript strings: {}.".format(
                text.get("group_column", "no group column"),
                fmt_num(text.get("duplicates_across_groups")),
                fmt_num(text["duplicate_values"]),
            ),
            "",
            "Samples:",
            "",
        ]
        for sample in text["samples"]:
            lines.append("- `{}`".format(sample.replace("`", "'")))
        lines += [""]

    if r.get("warnings"):
        lines += ["### Warnings / not measured", ""]
        lines += ["- {}".format(md_escape(w)) for w in r["warnings"]]
        lines += [""]

    return lines
