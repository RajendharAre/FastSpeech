"""Unit tests for scripts/plot_training_losses.py.

Run from the repository root:
    python -m unittest discover -s scripts/tests -v
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

from plot_training_losses import (  # noqa: E402
    LOSS_COLUMNS,
    LossCsvError,
    _step_segments,
    main,
    plot_losses,
    read_loss_csv,
)


def write_csv(path, rows, columns=LOSS_COLUMNS):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(str(value) for value in row) + "\n")
    return path


class ReadLossCsvTests(unittest.TestCase):
    def test_reads_every_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(
                os.path.join(tmp, "train_loss.csv"),
                [(100, 8.0, 4.0, 3.0, 1.0, 0.5, 0.2), (200, 7.0, 3.5, 2.5, 0.9, 0.4, 0.1)],
            )
            data = read_loss_csv(path)
            self.assertEqual(data["step"], [100, 200])
            self.assertEqual(data["total_loss"], [8.0, 7.0])
            self.assertEqual(data["duration_loss"], [0.2, 0.1])

    def test_missing_file(self):
        with self.assertRaises(LossCsvError):
            read_loss_csv(os.path.join(tempfile.gettempdir(), "definitely-absent.csv"))

    def test_missing_column_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(
                os.path.join(tmp, "x.csv"), [(100, 8.0)], columns=("step", "total_loss")
            )
            with self.assertRaises(LossCsvError) as ctx:
                read_loss_csv(path)
            self.assertIn("mel_loss", str(ctx.exception))

    def test_header_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(",".join(LOSS_COLUMNS) + "\n")
            with self.assertRaises(LossCsvError):
                read_loss_csv(path)

    def test_non_numeric_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(os.path.join(tmp, "x.csv"), [("100",) + ("nan-oops",) * 6])
            with self.assertRaises(LossCsvError):
                read_loss_csv(path)

    def test_out_of_order_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(
                os.path.join(tmp, "x.csv"),
                [(200, 7.0, 3.5, 2.5, 0.9, 0.4, 0.1), (100, 8.0, 4.0, 3.0, 1.0, 0.5, 0.2)],
            )
            with self.assertRaises(LossCsvError):
                read_loss_csv(path)

    def test_duplicate_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(
                os.path.join(tmp, "x.csv"),
                [
                    (100, 8.0, 4.0, 3.0, 1.0, 0.5, 0.2),
                    (100, 7.0, 3.5, 2.5, 0.9, 0.4, 0.1),
                ],
            )
            with self.assertRaisesRegex(LossCsvError, "strictly increasing"):
                read_loss_csv(path)


class PlotTests(unittest.TestCase):
    def _data(self):
        return {
            "step": [100, 200, 300],
            "total_loss": [8.0, 6.0, 5.0],
            "mel_loss": [4.0, 3.0, 2.5],
            "mel_postnet_loss": [3.5, 2.5, 2.0],
            "pitch_loss": [1.0, 0.8, 0.6],
            "energy_loss": [0.5, 0.4, 0.3],
            "duration_loss": [0.2, 0.15, 0.1],
        }

    def test_png_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "sub", "training_loss.png")
            plot_losses(self._data(), out, "Telugu training loss")
            self.assertTrue(os.path.isfile(out))
            self.assertGreater(os.path.getsize(out), 1000)

    def test_step_segments_leave_missing_interval_unbridged(self):
        self.assertEqual(
            _step_segments([100, 200, 300, 30100, 30200]),
            ([(0, 3), (3, 5)], [(300, 30100)]),
        )

    def test_plot_gaps_are_broken_and_component_legend_is_unique(self):
        data = {
            "step": [100, 200, 300, 30100, 30200],
            "total_loss": [8.0, 7.0, 6.0, 1.4, 1.3],
            "mel_loss": [4.0, 3.5, 3.0, 0.5, 0.4],
            "mel_postnet_loss": [3.5, 3.0, 2.5, 0.5, 0.4],
            "pitch_loss": [1.0, 0.9, 0.8, 0.1, 0.1],
            "energy_loss": [0.5, 0.4, 0.3, 0.05, 0.05],
            "duration_loss": [0.2, 0.15, 0.1, 0.03, 0.03],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "loss.png")
            with patch("plot_training_losses.plt.close") as close:
                plot_losses(data, out, "training loss")
                fig = close.call_args.args[0]

            total_lines = fig.axes[0].lines
            self.assertEqual(
                [list(line.get_xdata()) for line in total_lines],
                [[100, 200, 300], [30100, 30200]],
            )
            legend_labels = [
                text.get_text() for text in fig.axes[1].get_legend().get_texts()
            ]
            self.assertEqual(
                legend_labels,
                [
                    "mel loss",
                    "mel postnet loss",
                    "pitch loss",
                    "energy loss",
                    "duration loss",
                ],
            )
            self.assertTrue(os.path.isfile(out))
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_main_writes_both_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [(100, 8.0, 4.0, 3.0, 1.0, 0.5, 0.2), (200, 7.0, 3.5, 2.5, 0.9, 0.4, 0.1)]
            train = write_csv(os.path.join(tmp, "train_loss.csv"), rows)
            val = write_csv(os.path.join(tmp, "val_loss.csv"), rows)
            out_dir = os.path.join(tmp, "Results", "Telugu")
            main(["--train-csv", train, "--val-csv", val, "--out-dir", out_dir,
                 "--title", "Telugu"])
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "training_loss.png")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "validation_loss.png")))

    def test_main_requires_a_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                main(["--out-dir", tmp])


if __name__ == "__main__":
    unittest.main()
