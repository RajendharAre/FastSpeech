"""Kaggle-side export of the Svarah English corpus for the FastSpeech 2 pipeline.

Run on Kaggle (never locally; there is no Svarah data on the laptop):

    python scripts/export_svarah.py

Writes four things:

  English/data/raw/<speaker>/<clip>.wav|.lab   audio upsampled to 22,050 Hz
  /kaggle/working/mfa_corpus/english/<speaker>/  audio at the native 16 kHz plus
                                                 the aligner-ready transcript
  /kaggle/working/svarah_manifest/all.csv       clip_id,speaker,duration,transcript
  /kaggle/working/export_svarah_log.csv         one status row per clip

The manifest keeps the raw dataset text, so `scripts/split_dataset.py
--normalise-english` remains the only place normalisation is recorded. Svarah
ships no train/val split and no speaker labels, so the speaker directory name
comes from its `gender` column, which is the only usable label in the release.
"""
import csv
import io
import os
import sys
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from datasets import load_dataset, Audio

W = "/kaggle/working"
REPO = os.path.join(W, "FastSpeech")
MANIFEST = os.path.join(W, "svarah_manifest")
RAW = os.path.join(REPO, "English", "data", "raw")
MFA = os.path.join(W, "mfa_corpus", "english")
STATUS = os.path.join(W, "export_svarah_log.csv")

TARGET_SR = 22050
MFA_SR = 16000
# 980 mel frames at 22,050 Hz with hop 256: 1 + floor(samples / 256) <= 980.
MAX_SAMPLES = (980 - 1) * 256 + 255
MAX_SEC = MAX_SAMPLES / float(TARGET_SR)


def resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return np.asarray(x, dtype="float32")
    g = gcd(int(sr_in), int(sr_out))
    y = resample_poly(np.asarray(x, dtype="float64"), sr_out // g, sr_in // g)
    return np.clip(y, -1.0, 1.0).astype("float32")


def mfa_text(normaliser, text):
    """Aligner text: rule-normalised, lowercased, punctuation reduced to space.

    Apostrophes survive because the ARPA dictionary holds contracted entries
    (don't, she's); everything else that is not a letter or digit becomes a
    word boundary.
    """
    try:
        body, _rules = normaliser(text)
    except Exception:
        body = text
    out = []
    for ch in body.lower():
        if ch.isalnum() or ch == "'":
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def main():
    os.makedirs(MANIFEST, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(MFA, exist_ok=True)

    sys.path.insert(0, os.path.join(REPO, "English"))
    os.chdir(os.path.join(REPO, "English"))
    assert REPO in os.getcwd(), os.getcwd()
    from text.normalise import normalise_english

    print("normaliser module:", normalise_english.__module__, flush=True)

    ds = load_dataset("ai4bharat/Svarah", split="test")
    table = ds.data.table
    struct = table.column("audio_filepath").combine_chunks()
    print("rows:", len(struct), flush=True)

    durations = [float(v) for v in table.column("duration").to_pylist()]
    texts = table.column("text").to_pylist()
    genders = table.column("gender").to_pylist()

    ids = []
    seen = {}
    collisions = 0
    for row, item in enumerate(struct):
        path = (item.as_py() or {}).get("path") or ""
        stem = os.path.splitext(os.path.basename(path))[0] or "row%d" % row
        if stem in seen:
            collisions += 1
            stem = "%s-%d" % (stem, row)
        seen[stem] = True
        ids.append(stem)
    print("clip id collisions renamed:", collisions, flush=True)

    rows = []
    report = []
    skipped = []
    exported = failed = too_long = empty_text = 0

    for row in range(len(struct)):
        clip_id = ids[row]
        speaker = (genders[row] or "unknown").strip().lower()
        text = (texts[row] or "").strip()
        value = struct[row].as_py()
        payload = (value or {}).get("bytes")
        if not payload:
            failed += 1
            report.append((clip_id, "no_bytes", ""))
            continue
        if not text:
            empty_text += 1
        try:
            audio, sr = sf.read(io.BytesIO(payload), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            seconds = len(audio) / float(sr)
            # The budget is 22,050 Hz mel frames, so the source-rate sample
            # count must never be compared against MAX_SAMPLES directly.
            if seconds > MAX_SEC:
                too_long += 1
                skipped.append((clip_id, round(seconds, 3)))
                report.append((clip_id, "too_long", "%.3f s" % seconds))
                continue

            os.makedirs(os.path.join(RAW, speaker), exist_ok=True)
            os.makedirs(os.path.join(MFA, speaker), exist_ok=True)

            high = resample(audio, sr, TARGET_SR)
            if high.shape[0] > MAX_SAMPLES:
                raise AssertionError(
                    "exported wav has %d samples, over the %d-sample budget"
                    % (high.shape[0], MAX_SAMPLES)
                )
            sf.write(os.path.join(RAW, speaker, clip_id + ".wav"), high,
                     TARGET_SR, subtype="PCM_16")
            with open(os.path.join(RAW, speaker, clip_id + ".lab"), "w",
                      encoding="utf-8") as f:
                f.write(text + "\n")

            low = resample(audio, sr, MFA_SR)
            sf.write(os.path.join(MFA, speaker, clip_id + ".wav"), low, MFA_SR,
                     subtype="PCM_16")
            with open(os.path.join(MFA, speaker, clip_id + ".lab"), "w",
                      encoding="utf-8") as f:
                f.write(mfa_text(normalise_english, text) + "\n")

            rows.append({
                "clip_id": clip_id,
                "speaker": speaker,
                "duration": "%.6f" % seconds,
                "transcript": text,
            })
            report.append((clip_id, "ok", "%.3f s" % seconds))
            exported += 1
            if exported % 250 == 0:
                print("exported", exported, "too_long", too_long,
                      "failed", failed, flush=True)
        except Exception as exc:
            failed += 1
            report.append((clip_id, "error", repr(exc)))
            print("FAIL", clip_id, repr(exc), flush=True)

    columns = ("clip_id", "speaker", "duration", "transcript")
    with open(os.path.join(MANIFEST, "all.csv"), "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    with open(STATUS, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "status", "detail"])
        writer.writerows(report)

    hours = sum(float(r["duration"]) for r in rows) / 3600.0
    per_speaker = {}
    for r in rows:
        per_speaker[r["speaker"]] = per_speaker.get(r["speaker"], 0.0) + float(r["duration"]) / 3600.0
    print("DONE exported", exported, "hours %.4f" % hours, flush=True)
    print("speakers:", {k: round(v, 4) for k, v in sorted(per_speaker.items())}, flush=True)
    print("too_long", too_long, "failed", failed, "empty_text", empty_text, flush=True)
    print("longest exported clip %.3f s, guard %.3f s" % (
        max([float(r["duration"]) for r in rows]) if rows else 0.0, MAX_SEC), flush=True)
    with open(os.path.join(MANIFEST, "too_long.csv"), "w", newline="",
              encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "duration"])
        writer.writerows(skipped)


if __name__ == "__main__":
    main()
