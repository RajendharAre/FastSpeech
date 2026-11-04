"""Render the training and validation loss plots required by the deliverables.

``train.py`` appends loss rows to ``logs/<Lang>/train/<step>/train_loss.csv`` and
``logs/<Lang>/val/<step>/val_loss.csv`` (``utils/train_utils.append_loss_csv``),
but nothing in the repository drew them. This script turns those CSVs into
``Results/<Lang>/training_loss.png`` and ``validation_loss.png``.
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

LOSS_COLUMNS = (
    "step",
    "total_loss",
    "mel_loss",
    "mel_postnet_loss",
    "pitch_loss",
    "energy_loss",
    "duration_loss",
)
COMPONENTS = LOSS_COLUMNS[2:]
COMPONENT_COLORS = {
    "mel_loss": "tab:blue",
    "mel_postnet_loss": "tab:cyan",
    "pitch_loss": "tab:red",
    "energy_loss": "tab:purple",
    "duration_loss": "tab:green",
}


class LossCsvError(ValueError):
    """Raised when a loss CSV is missing, malformed or empty."""


def read_loss_csv(path):
    """Return ``{"step": [...], "<loss>": [...]}`` from a loss CSV."""
    if not os.path.isfile(path):
        raise LossCsvError("loss CSV not found: {}".format(path))

    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise LossCsvError("{}: empty file (no header)".format(path))
        missing = [column for column in LOSS_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise LossCsvError(
                "{}: missing column(s): {}".format(path, ", ".join(missing))
            )
        raw_rows = list(reader)

    if not raw_rows:
        raise LossCsvError("{}: header only, no loss rows".format(path))

    data = {column: [] for column in LOSS_COLUMNS}
    for line_number, raw in enumerate(raw_rows, start=2):
        try:
            step = int(float(raw["step"]))
            values = [float(raw[column]) for column in LOSS_COLUMNS[1:]]
        except (TypeError, ValueError):
            raise LossCsvError(
                "{}:{}: step and losses must be numbers".format(path, line_number)
            )
        data["step"].append(step)
        for column, value in zip(LOSS_COLUMNS[1:], values):
            data[column].append(value)

    if data["step"] != sorted(data["step"]):
        raise LossCsvError(
            "{}: steps are not monotonically increasing; the CSV looks like two "
            "runs appended together".format(path)
        )
    return data


def plot_losses(data, out_path, title):
    """Draw total loss and the five component losses into ``out_path``."""
    steps = data["step"]
    fig, (ax_total, ax_components) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax_total.plot(steps, data["total_loss"], color="black", linewidth=1.5)
    ax_total.set_ylabel("total loss")
    ax_total.set_title(title)
    ax_total.grid(alpha=0.3)

    for column in COMPONENTS:
        ax_components.plot(
            steps,
            data[column],
            color=COMPONENT_COLORS[column],
            linewidth=1.2,
            label=column.replace("_", " "),
        )
    ax_components.set_yscale("log")
    ax_components.set_ylabel("component loss (log)")
    ax_components.set_xlabel("training step")
    ax_components.grid(alpha=0.3)
    ax_components.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-csv", help="train_loss.csv written by train.py")
    parser.add_argument("--val-csv", help="val_loss.csv written by evaluate()")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="directory for training_loss.png / validation_loss.png",
    )
    parser.add_argument("--title", default="", help="language label used in the titles")
    args = parser.parse_args(argv)

    if not args.train_csv and not args.val_csv:
        raise SystemExit("pass --train-csv and/or --val-csv")

    prefix = "{} ".format(args.title) if args.title else ""
    written = []
    if args.train_csv:
        written.append(
            plot_losses(
                read_loss_csv(args.train_csv),
                os.path.join(args.out_dir, "training_loss.png"),
                "{}training loss".format(prefix),
            )
        )
    if args.val_csv:
        written.append(
            plot_losses(
                read_loss_csv(args.val_csv),
                os.path.join(args.out_dir, "validation_loss.png"),
                "{}validation loss".format(prefix),
            )
        )
    for path in written:
        print("wrote {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
