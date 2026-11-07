"""Compare two decoded Telugu checkpoint exports on the numbers in their JSON payloads."""
import glob
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(ROOT, "Results", "Telugu")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EARLY_EXPORT_ROOT = os.path.join(
    os.path.dirname(PROJECT_ROOT), "FastSpeech_Data", "Telugu", "checkpoint_5000"
)
HOP = 256
SR = 22050
GT_FRAMES_PER_SYMBOL = 6.25
GT_PITCH_STD = 0.941
GT_ENERGY_STD = 0.948


def load(step):
    if step == 60000:
        base = os.path.join(ROOT, "variance_adaptor_plots")
    elif step == 5000:
        base = os.path.join(EARLY_EXPORT_ROOT, "variance_adaptor_plots")
    else:
        raise ValueError("unsupported comparison checkpoint: {}".format(step))
    if not os.path.isdir(base):
        raise FileNotFoundError(
            "variance export for checkpoint {} not found: {}".format(step, base)
        )
    out = {}
    for path in sorted(glob.glob(os.path.join(base, "T*.json"))):
        with open(path, encoding="utf-8") as f:
            out[os.path.basename(path)[:-5]] = json.load(f)
    if not out:
        raise FileNotFoundError("no Telugu prediction JSON files found in {}".format(base))
    return out


def summarise(step, rows):
    print("\n=== checkpoint {} ===".format(step))
    print("{:<5s} {:>6s} {:>6s} {:>7s} {:>7s} {:>7s} {:>9s} {:>9s}".format(
        "id", "sym", "frames", "sumdur", "sec", "fr/sym", "pitch_std", "energy_std"))
    fps, pstd, estd, secs = [], [], [], []
    for key, d in rows.items():
        sym = len(d["symbols"])
        frames = int(d["mel_frames"])
        total = float(np.sum(d["duration_frames"]))
        sec = frames * HOP / SR
        f = frames / sym
        p = float(np.std(d["pitch_normalized"]))
        e = float(np.std(d["energy_normalized"]))
        fps.append(f)
        pstd.append(p)
        estd.append(e)
        secs.append(sec)
        print("{:<5s} {:6d} {:6d} {:7.1f} {:7.2f} {:7.2f} {:9.3f} {:9.3f}".format(
            key, sym, frames, total, sec, f, p, e))
    print("{:<5s} {:>32s} {:7.2f} {:9.3f} {:9.3f}".format(
        "mean", "", float(np.mean(fps)), float(np.mean(pstd)), float(np.mean(estd))))
    return dict(frames_per_symbol=float(np.mean(fps)), pitch_std=float(np.mean(pstd)),
                energy_std=float(np.mean(estd)), seconds=dict(zip(rows, secs)))


def main():
    steps = (5000, 60000)
    got = {}
    for step in steps:
        got[step] = summarise(step, load(step))
    print("\n=== summary against the corpus ===")
    print("ground truth: {:.2f} frames/symbol, pitch std {:.3f}, energy std {:.3f}".format(
        GT_FRAMES_PER_SYMBOL, GT_PITCH_STD, GT_ENERGY_STD))
    print("{:<8s} {:>10s} {:>10s} {:>11s}".format("checkpoint", "fr/symbol", "pitch_std", "energy_std"))
    for step in steps:
        g = got[step]
        print("{:<8d} {:10.2f} {:10.3f} {:11.3f}".format(
            step, g["frames_per_symbol"], g["pitch_std"], g["energy_std"]))
    print("\n=== per-sentence duration, seconds ===")
    print("{:<8s}".format("checkpoint") + "".join("{:>8s}".format(k) for k in got[steps[0]]["seconds"]))
    for step in steps:
        print("{:<8d}".format(step) + "".join(
            "{:8.2f}".format(v) for v in got[step]["seconds"].values()))


if __name__ == "__main__":
    main()
