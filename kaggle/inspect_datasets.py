#!/usr/bin/env python3
"""Inspect a Hugging Face audio dataset and write docs/dataset_report.md.

Designed for Kaggle (Linux). It does NOT assume column names:

* every config and split is enumerated through the Hugging Face API;
* every column and feature type is printed and carried into the report;
* the audio column is found by feature type (`Audio`), not by name;
* durations and sampling rates are read from encoded audio headers without
  decoding the waveforms (fast, and exact for each clip);
* every low-cardinality column (speaker ids, gender, split, ...) is reported as a
  group table with clip counts and hours, so the human can decide which column is
  the speaker column;
* licence and gating come from the Hugging Face API and are labelled as such.

Usage (one dataset per session, then let the script rebuild the combined report):

    python kaggle/inspect_datasets.py --dataset svarah
    python kaggle/inspect_datasets.py --dataset indictts_telugu
    python kaggle/inspect_datasets.py --rebuild-only

The token is read from the HF_TOKEN environment variable (Kaggle Secret). It is
never printed or written to disk.
"""
import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_utils import (  # noqa: E402
    duplicate_across_groups,
    duration_stats,
    group_stats,
    probe_audio_value,
    render_markdown,
    text_stats,
)

DATASETS = {
    "svarah": {"repo_id": "ai4bharat/Svarah", "language": "English"},
    "indictts_telugu": {
        "repo_id": "SPRINGLab/IndicTTS_Telugu",
        "language": "Telugu",
    },
}

MAX_GROUP_VALUES = 200
MAX_TEXT_LENGTH_FOR_GROUPING = 60


def versions():
    out = {"python_version": platform.python_version()}
    for key, module in (
        ("datasets_version", "datasets"),
        ("hub_version", "huggingface_hub"),
        ("soundfile_version", "soundfile"),
    ):
        try:
            out[key] = __import__(module).__version__
        except Exception:
            out[key] = "unavailable"
    return out


def hf_metadata(repo_id, token):
    """Licence and gating fields, quoted from the Hub API."""
    meta = {"repo_id": repo_id, "card_error": None}
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).dataset_info(repo_id)
        card = info.cardData or {}
        meta.update(
            {
                "sha": info.sha,
                "gated": info.gated,
                "private": info.private,
                "last_modified": str(info.lastModified),
                "license": card.get("license"),
                "tags": [
                    t
                    for t in (info.tags or [])
                    if t.startswith(
                        ("license:", "language:", "task_categories:", "size_categories:")
                    )
                ],
            }
        )
    except Exception as e:  # gated repo without accepted terms, network, etc.
        meta["card_error"] = "{}: {}".format(type(e).__name__, e)
    return meta


def audio_feature_columns(features):
    try:
        from datasets import Audio
    except ImportError:  # older/newer layouts
        from datasets.features import Audio

    # isinstance() is the primary check; the class-name check keeps this working
    # if the feature class is re-exported differently in a future datasets release.
    return [
        name
        for name, feature in features.items()
        if isinstance(feature, Audio) or type(feature).__name__ == "Audio"
    ]


def label_columns(features):
    """ClassLabel columns (for example gender) and their class names."""
    from datasets import ClassLabel

    return {
        name: list(feature.names)
        for name, feature in features.items()
        if isinstance(feature, ClassLabel)
    }


def scalar_columns(features):
    """Columns that are safe to group by: strings, labels, ints, floats, bools."""
    from datasets import ClassLabel, Value

    allowed = {"string", "int8", "int16", "int32", "int64", "float32", "float64", "bool"}
    names = []
    for name, feature in features.items():
        if isinstance(feature, ClassLabel):
            names.append(name)
        elif isinstance(feature, Value) and str(feature.dtype) in allowed:
            names.append(name)
    return names


def cast_audio_to_bytes(dataset, columns):
    from datasets import Audio

    for column in columns:
        try:
            dataset = dataset.cast_column(column, Audio(decode=False))
        except Exception as e:
            print("  WARNING: cannot disable decoding for '{}': {}".format(column, e))
    return dataset


def inspect(args):
    from datasets import (
        Value,
        get_dataset_config_names,
        get_dataset_split_names,
        load_dataset,
    )

    spec = DATASETS[args.dataset]
    repo_id = spec["repo_id"]
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print(
            "WARNING: HF_TOKEN is not set. This works for open datasets, but "
            "ai4bharat/Svarah is gated and will fail."
        )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    meta = hf_metadata(repo_id, token)
    meta.update(versions())
    meta["language"] = spec["language"]

    if meta.get("card_error"):
        meta["licence_note"] = (
            "Licence/gating could not be read from the Hub API. Check the dataset "
            "card manually before relying on any licence claim."
        )

    print("Dataset: {} (revision {})".format(repo_id, meta.get("sha")))
    print("Gated: {} | licence (card): {}".format(meta.get("gated"), meta.get("license")))

    try:
        configs = get_dataset_config_names(repo_id, token=token) or ["default"]
    except Exception as e:
        raise SystemExit(
            "Could not list configs for {}: {}\nIf the dataset is gated, accept "
            "the conditions on the dataset page with the same account that owns "
            "HF_TOKEN, then rerun.".format(repo_id, e)
        )
    print("Configs:", configs)
    split_rows = []
    audio_measurements = {}  # column -> list of {frames, samplerate, format}
    audio_failures = {}  # column -> list of error strings
    col_values = {}  # column -> list of values (aggregated across splits)
    string_columns = []
    numeric_duration_columns = []
    warnings = []
    total_rows = 0
    aligned_durations = []  # one entry per row; None when the header failed
    primary_audio = None
    label_names = {}

    for config in configs:
        try:
            splits = get_dataset_split_names(repo_id, config, token=token)
        except Exception as e:
            warnings.append(
                "Could not list splits for config '{}': {}".format(config, e)
            )
            splits = [None]
        if not splits:
            splits = [None]

        for split in splits:
            print("Loading config={} split={} ...".format(config, split))
            try:
                dataset = load_dataset(
                    repo_id, config, split=split, token=token
                )
            except Exception as e:
                warnings.append(
                    "Could not load config='{}' split='{}': {} (if this is a "
                    "gated dataset, log in and accept the conditions with the same "
                    "account as the token)".format(config, split, e)
                )
                print("  FAILED:", e)
                continue

            n = len(dataset)
            total_rows += n
            features = dataset.features
            split_rows.append(
                {
                    "config": config,
                    "split": split,
                    "num_rows": n,
                    "features": str(features),
                }
            )
            print("  rows: {:,}".format(n))

            audio_cols = audio_feature_columns(features)
            label_names.update(label_columns(features))
            if primary_audio is None and audio_cols:
                primary_audio = audio_cols[0]
            elif audio_cols and primary_audio not in audio_cols:
                warnings.append(
                    "Split {}/{} has different audio columns ({}); the primary "
                    "audio column '{}' is not present.".format(
                        config, split, audio_cols, primary_audio
                    )
                )

            # String columns are identified from the declared feature types,
            # not from the first row's Python value.
            for name, feature in features.items():
                if (
                    isinstance(feature, Value)
                    and str(feature.dtype) == "string"
                    and name not in string_columns
                ):
                    string_columns.append(name)

            dataset = cast_audio_to_bytes(dataset, audio_cols)

            scalar_cols = scalar_columns(features)
            for column in scalar_cols:
                col_values.setdefault(column, [])
            for column in audio_cols:
                audio_measurements.setdefault(column, [])
                audio_failures.setdefault(column, [])

            for i, row in enumerate(dataset):
                for column in scalar_cols:
                    col_values[column].append(row.get(column))
                row_duration = None
                for column in audio_cols:
                    probe = probe_audio_value(row.get(column))
                    if "error" in probe:
                        audio_failures[column].append(probe["error"])
                    else:
                        audio_measurements[column].append(probe)
                        if column == primary_audio and probe.get("samplerate"):
                            row_duration = probe["frames"] / float(probe["samplerate"])
                aligned_durations.append(row_duration)
                if (i + 1) % 1000 == 0:
                    print("  {} rows processed".format(i + 1))

            # Columns such as "duration" are reported as metadata, not as truth.
            for name, feature in features.items():
                if (
                    isinstance(feature, Value)
                    and "dur" in name.lower()
                    and str(feature.dtype).startswith(("float", "int"))
                ):
                    numeric_duration_columns.append(name)

    # ---- audio summary -------------------------------------------------
    audio_out = {
        "columns": [],
        "duration_stats": None,
        "primary_column": primary_audio,
    }
    primary_durations = [d for d in aligned_durations if d is not None]
    for column, measurements in audio_measurements.items():
        rates = {}
        durations = []
        for m in measurements:
            rate = m["samplerate"]
            rates[str(rate)] = rates.get(str(rate), 0) + 1
            if rate:
                durations.append(m["frames"] / float(rate))
        audio_out["columns"].append(
            {
                "name": column,
                "measured": len(measurements),
                "failed": len(audio_failures.get(column, [])),
                "sampling_rates": rates,
                "total_hours": sum(durations) / 3600.0 if durations else None,
                "failure_examples": audio_failures.get(column, [])[:3],
            }
        )
    audio_out["duration_stats"] = duration_stats(primary_durations) if primary_durations else None

    if numeric_duration_columns:
        out = []
        for column in sorted(set(numeric_duration_columns)):
            values = [
                float(v)
                for v in col_values.get(column, [])
                if isinstance(v, (int, float))
            ]
            if values:
                out.append(
                    {
                        "name": column,
                        "n": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                    }
                )
        audio_out["metadata_duration_columns"] = out

    # ---- text summary --------------------------------------------------
    text_out = []
    text_by_column = {}
    for column in string_columns:
        values = col_values.get(column, [])
        stats = text_stats(values)
        text_out.append({"column": column, **stats})
        text_by_column[column] = stats

    # ---- group summary -------------------------------------------------
    group_columns = []
    for column, values in col_values.items():
        if column in string_columns:
            stats = text_by_column.get(column, {})
            if (
                stats.get("distinct_values", 0) > MAX_GROUP_VALUES
                or stats.get("length_mean", 0.0) > MAX_TEXT_LENGTH_FOR_GROUPING
            ):
                continue
        distinct = len({str(v) for v in values})
        if 2 <= distinct <= MAX_GROUP_VALUES:
            group_columns.append((column, distinct))

    groups_out = []
    for column, distinct in sorted(group_columns, key=lambda kv: -kv[1]):
        values = col_values[column]
        names = label_names.get(column)
        if names:  # show class names instead of raw indices
            values = [
                names[v] if isinstance(v, int) and 0 <= v < len(names) else v
                for v in values
            ]
        table = group_stats(values, aligned_durations)
        csv_path = None
        if len(table) > 30:
            csv_path = "docs/dataset_report_{}_groups_{}.csv".format(
                args.dataset, column
            )
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["value", "clips", "hours"])
                for entry in table:
                    writer.writerow(
                        [entry["value"], entry["clips"], "{:.6f}".format(entry["hours"])]
                    )
        groups_out.append(
            {
                "column": column,
                "distinct": distinct,
                "top": table[:30],
                "csv_path": csv_path,
                "clips_without_duration": sum(e["missing"] for e in table),
            }
        )

    # Best available group column for the cross-group duplicate check.
    # Speaker-like names win, then the highest-cardinality low-cardinality column.
    group_column = None
    if groups_out:
        def priority(group):
            name = group["column"].lower()
            speaker_like = any(
                key in name for key in ("speaker", "spk", "gender", "sex", "voice")
            )
            return (0 if speaker_like else 1, group["distinct"])

        group_column = sorted(groups_out, key=priority)[0]["column"]
    if group_column:
        for entry in text_out:
            entry["group_column"] = group_column
            entry["duplicates_across_groups"] = duplicate_across_groups(
                col_values.get(entry["column"], []), col_values.get(group_column, [])
            )

    if not primary_durations:
        warnings.append(
            "No clip duration could be measured from an audio header. Total hours "
            "are not reported. Do not substitute a value from the dataset card."
        )
    for column, failures in audio_failures.items():
        if failures:
            warnings.append(
                "{} audio clip(s) could not be read in column '{}' (first error: {})".format(
                    len(failures), column, failures[0]
                )
            )
    if total_rows and not text_out:
        warnings.append("No string column was detected; transcripts were not inspected.")

    summary = {
        "rows": total_rows,
        "splits_loaded": len(split_rows),
        "clips_measured": len(primary_durations),
        "total_hours": (sum(primary_durations) / 3600.0)
        if primary_durations
        else None,
        "audio_failures": sum(len(v) for v in audio_failures.values()),
        "configs": configs,
        "group_column": group_column,
    }

    report = {
        "meta": meta,
        "generated_at": now,
        "command": " ".join(sys.argv),
        "splits": split_rows,
        "audio": audio_out,
        "groups": groups_out,
        "text": text_out,
        "summary": summary,
        "label_columns": label_names,
        "warnings": warnings,
        "json_path": "docs/dataset_report_{}.json".format(args.dataset),
    }

    with open(report["json_path"], "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Wrote", report["json_path"])

    print("\nSUMMARY")
    print("  rows:", summary["rows"])
    print("  clips measured:", summary["clips_measured"])
    print(
        "  total hours:",
        "not measured"
        if summary["total_hours"] is None
        else "{:.4f}".format(summary["total_hours"]),
    )
    print("  audio failures:", summary["audio_failures"])
    if warnings:
        print("  warnings:")
        for w in warnings:
            print("   -", w)


def rebuild(out_md):
    paths = sorted(Path("docs").glob("dataset_report_*.json"))
    if not paths:
        raise SystemExit("No docs/dataset_report_*.json found; run an inspection first.")
    reports = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        report["json_path"] = str(path).replace("\\", "/")
        reports.append(report)
    text = render_markdown(reports, reports[0]["generated_at"])
    Path(out_md).write_text(text, encoding="utf-8")
    print("Wrote {} from {} report(s)".format(out_md, len(reports)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    parser.add_argument("--token", default=None, help="defaults to $HF_TOKEN")
    parser.add_argument("--out", default="docs/dataset_report.md")
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="only rebuild docs/dataset_report.md from existing JSON reports",
    )
    args = parser.parse_args()

    if not args.rebuild_only:
        if not args.dataset:
            parser.error("--dataset is required unless --rebuild-only is used")
        inspect(args)
    rebuild(args.out)
