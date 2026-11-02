"""Preprocessing adapted from ming024/FastSpeech2 (MIT).

MODIFIED(both):
* selection comes from manifests written by ``scripts/split_dataset.py``;
  the reference random shuffle/``val_size`` split was removed;
* a clip that cannot be processed is logged with a reason, and the run aborts
  before normalisation so a rerun cannot normalise features twice;
* audio is loaded CPU-side with an explicit sampling rate;
* output order is deterministic (manifest order, sorted speakers).
"""
import csv
import json
import os

import librosa
import numpy as np
import pyworld as pw
import tgt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio
from preprocessor.manifest import load_manifests
from preprocessor.report import (
    LOG_COLUMNS,
    ClipSkipped,
    format_summary,
    summarize,
    write_summary,
)
from text import text_to_sequence


class Preprocessor:
    def __init__(self, config, allow_dropped=False):
        self.config = config
        self.allow_dropped = allow_dropped
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.manifest_paths = {
            "train": config["path"]["manifest_train"],
            "val": config["path"]["manifest_val"],
        }
        self.cleaners = config["preprocessing"]["text"]["text_cleaners"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        entries = load_manifests(self.manifest_paths)
        print("Processing Data ...")

        log_rows = []
        outputs = {}
        pitch_paths = []
        energy_paths = []
        n_frames = 0
        processed_by_split = {"train": 0, "val": 0}
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        for split, speaker, basename in tqdm(entries, desc="Clips"):
            try:
                info, pitch, energy, frames = self.process_utterance(
                    speaker, basename
                )
            except ClipSkipped as exc:
                reason = " ".join(exc.detail.split())
                log_rows.append(
                    {
                        "speaker": speaker,
                        "clip": basename,
                        "split": split,
                        "status": "failed",
                        "category": exc.category,
                        "reason": reason,
                    }
                )
                tqdm.write(
                    "DROPPED {}/{} [{}] {}: {}".format(
                        speaker, basename, split, exc.category, reason
                    )
                )
                continue

            outputs[(split, speaker, basename)] = info
            log_rows.append(
                {
                    "speaker": speaker,
                    "clip": basename,
                    "split": split,
                    "status": "ok",
                    "category": "",
                    "reason": "",
                }
            )
            processed_by_split[split] += 1
            pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
            energy_scaler.partial_fit(energy.reshape((-1, 1)))
            pitch_paths.append(
                os.path.join(self.out_dir, "pitch", "{}-pitch-{}.npy".format(speaker, basename))
            )
            energy_paths.append(
                os.path.join(self.out_dir, "energy", "{}-energy-{}.npy".format(speaker, basename))
            )
            n_frames += frames

        log_path = os.path.join(self.out_dir, "preprocessing_log.csv")
        self._write_log(log_path, log_rows)
        summary = summarize(log_rows)
        summary_path = os.path.join(self.out_dir, "preprocessing_summary.json")
        write_summary(summary_path, summary)
        print(format_summary(summary))
        print("Preprocessing summary: {}".format(summary_path))

        if summary["dropped"] and not self.allow_dropped:
            raise RuntimeError(
                "{} of {} manifest clips were dropped; see {} and {}. Fix the "
                "clips or rerun with --allow-dropped to continue without them. "
                "No train.txt/val.txt, stats.json or normalised features were "
                "written.".format(
                    summary["dropped"], summary["total"], log_path, summary_path
                )
            )
        if summary["dropped"]:
            print(
                "WARNING: continuing with {} usable clips; {} dropped clip(s) "
                "are listed in {}".format(
                    summary["ok"], summary["dropped"], log_path
                )
            )

        empty_splits = [
            split for split in ("train", "val") if processed_by_split[split] == 0
        ]
        if empty_splits:
            raise RuntimeError(
                "no usable clips in split(s): {}; see {}".format(
                    ", ".join(empty_splits), log_path
                )
            )

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = float(pitch_scaler.mean_[0])
            pitch_std = float(pitch_scaler.scale_[0])
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = float(energy_scaler.mean_[0])
            energy_std = float(energy_scaler.scale_[0])
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(pitch_paths, pitch_mean, pitch_std)
        energy_min, energy_max = self.normalize(energy_paths, energy_mean, energy_std)

        # Save files. Speaker IDs are assigned from a sorted list so a rerun
        # (or a different filesystem order) produces the same mapping. Only
        # speakers with at least one usable clip are included.
        speakers = sorted({speaker for _split, speaker, _clip in outputs})
        speaker_map = {speaker: index for index, speaker in enumerate(speakers)}
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speaker_map, sort_keys=True))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        # Write metadata in manifest order. A missing entry is either an
        # internal error (strict mode, where any drop aborted earlier) or a
        # clip the caller explicitly allowed us to drop.
        written = {"train": 0, "val": 0}
        for split in ("train", "val"):
            with open(
                os.path.join(self.out_dir, "{}.txt".format(split)),
                "w",
                encoding="utf-8",
            ) as f:
                for entry_split, speaker, basename in entries:
                    if entry_split != split:
                        continue
                    info = outputs.get((split, speaker, basename))
                    if info is None:
                        if self.allow_dropped:
                            continue
                        raise RuntimeError(
                            "internal error: no output for {}/{}".format(
                                speaker, basename
                            )
                        )
                    f.write(info + "\n")
                    written[split] += 1

        print(
            "Processed {} clips ({} train, {} val), {} speakers; {} dropped".format(
                summary["ok"],
                written["train"],
                written["val"],
                len(speaker_map),
                summary["dropped"],
            )
        )
        print(
            "Total time: {:.3f} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )
        print("Preprocessing log: {}".format(log_path))

        return list(outputs.values())

    @staticmethod
    def _write_log(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=LOG_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        for label, path in (
            ("audio", wav_path),
            ("transcript", text_path),
            ("TextGrid", tg_path),
        ):
            if not os.path.isfile(path):
                raise ClipSkipped(
                    "missing {}".format(label),
                    "missing {} file: {}".format(label, path),
                )

        # Get alignments
        try:
            textgrid = tgt.io.read_textgrid(tg_path)
        except Exception as exc:
            raise ClipSkipped("unreadable TextGrid", "cannot read TextGrid: {}".format(exc))
        try:
            tier = textgrid.get_tier_by_name("phones")
        except Exception as exc:
            raise ClipSkipped(
                "TextGrid structure", "TextGrid has no 'phones' tier: {}".format(exc)
            )

        phone, duration, start, end = self.get_alignment(tier)
        if not phone:
            raise ClipSkipped(
                "alignment", "no phones after trimming leading/trailing silence"
            )
        if start >= end:
            raise ClipSkipped(
                "alignment",
                "invalid alignment span: start {:.3f}s >= end {:.3f}s".format(
                    start, end
                ),
            )
        if any("|" in label for label in phone):
            raise ClipSkipped(
                "alignment", "a phone label contains the metadata separator '|'"
            )
        if any(value <= 0 for value in duration):
            raise ClipSkipped(
                "alignment", "alignment contains a non-positive phone duration"
            )

        n_alignment_frames = int(sum(duration))
        if n_alignment_frames <= 0:
            raise ClipSkipped("alignment", "alignment has zero total duration")

        text = "{" + " ".join(phone) + "}"
        try:
            encoded = text_to_sequence(text, self.cleaners)
        except Exception as exc:
            raise ClipSkipped(
                "text frontend",
                "alignment phones cannot be encoded by the configured frontend: "
                "{}".format(exc),
            )
        if len(encoded) != len(duration):
            raise ClipSkipped(
                "text frontend",
                "alignment encodes to {} symbol(s) but has {} phone duration(s); "
                "text and duration lengths must match".format(
                    len(encoded), len(duration)
                ),
            )

        # Read and trim wav files. librosa resamples explicitly to the rate the
        # mel frontend and HiFi-GAN expect; nothing assumes CUDA.
        try:
            wav, _ = librosa.load(wav_path, sr=self.sampling_rate, mono=True)
        except Exception as exc:
            raise ClipSkipped("audio read", "cannot read audio: {}".format(exc))
        if wav.size == 0:
            raise ClipSkipped("audio read", "audio is empty")
        if not np.isfinite(wav).all():
            raise ClipSkipped("audio content", "audio contains non-finite samples")
        wav = np.clip(wav, -1.0, 1.0).astype(np.float32)

        start_sample = int(self.sampling_rate * start)
        end_sample = int(self.sampling_rate * end)
        if start_sample < 0 or start_sample >= len(wav):
            raise ClipSkipped(
                "audio/alignment",
                "alignment starts at {:.3f}s but audio is {:.3f}s long".format(
                    start, len(wav) / self.sampling_rate
                ),
            )
        if end_sample > len(wav):
            raise ClipSkipped(
                "audio/alignment",
                "alignment ends at {:.3f}s but audio is {:.3f}s long".format(
                    end, len(wav) / self.sampling_rate
                ),
            )
        wav = wav[start_sample:end_sample].astype(np.float32)
        if wav.size == 0:
            raise ClipSkipped("audio/alignment", "trimmed audio is empty")
        min_samples = self.STFT.stft_fn.filter_length // 2 + 1
        if wav.size < min_samples:
            raise ClipSkipped(
                "audio too short",
                "trimmed audio has {} sample(s); the STFT needs at least {}".format(
                    wav.size, min_samples
                ),
            )

        # Read raw text
        try:
            with open(text_path, "r", encoding="utf-8-sig") as f:
                lines = [line.strip() for line in f.read().splitlines()]
        except Exception as exc:
            raise ClipSkipped("transcript", "cannot read transcript: {}".format(exc))
        if not lines or not any(lines):
            raise ClipSkipped("transcript", "transcript file is empty")
        if len(lines) != 1:
            raise ClipSkipped(
                "transcript",
                "transcript contains {} lines; expected exactly one".format(len(lines)),
            )
        raw_text = lines[0]
        if "|" in raw_text:
            raise ClipSkipped(
                "transcript", "transcript contains the metadata separator '|'"
            )

        # Compute fundamental frequency
        try:
            pitch, t = pw.dio(
                wav.astype(np.float64),
                self.sampling_rate,
                frame_period=self.hop_length / self.sampling_rate * 1000,
            )
            pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        except Exception as exc:
            raise ClipSkipped(
                "pitch extraction", "pyworld pitch extraction failed: {}".format(exc)
            )

        if len(pitch) < n_alignment_frames:
            raise ClipSkipped(
                "pitch extraction",
                "pitch has {} frame(s) but the alignment needs {}".format(
                    len(pitch), n_alignment_frames
                ),
            )
        pitch = pitch[:n_alignment_frames]
        voiced = int(np.sum(pitch != 0))
        if voiced <= 1:
            raise ClipSkipped(
                "pitch extraction",
                "only {} voiced pitch frame(s); need at least 2".format(voiced),
            )

        # Compute mel-scale spectrogram and energy
        try:
            mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        except Exception as exc:
            raise ClipSkipped(
                "feature extraction",
                "mel-spectrogram extraction failed: {}".format(exc),
            )
        if mel_spectrogram.shape[1] < n_alignment_frames:
            raise ClipSkipped(
                "feature extraction",
                "mel-spectrogram has {} frame(s) but the alignment needs {}".format(
                    mel_spectrogram.shape[1], n_alignment_frames
                ),
            )
        if len(energy) < n_alignment_frames:
            raise ClipSkipped(
                "feature extraction",
                "energy has {} frame(s) but the alignment needs {}".format(
                    len(energy), n_alignment_frames
                ),
            )
        mel_spectrogram = mel_spectrogram[:, :n_alignment_frames]
        energy = energy[:n_alignment_frames]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        pitch_for_stats = self.remove_outlier(pitch)
        energy_for_stats = self.remove_outlier(energy)
        if pitch_for_stats.size == 0:
            raise ClipSkipped(
                "feature statistics", "pitch outlier filter removed every frame"
            )
        if energy_for_stats.size == 0:
            raise ClipSkipped(
                "feature statistics", "energy outlier filter removed every frame"
            )

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            pitch_for_stats,
            energy_for_stats,
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        intervals = getattr(tier, "intervals", None)
        if intervals is None:
            intervals = getattr(tier, "_objects", None)
        if intervals is None:
            raise ClipSkipped(
                "TextGrid structure", "phones tier has no interval objects"
            )

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in intervals:
            s, e, p = t.start_time, t.end_time, t.text.strip()
            if not p:
                raise ClipSkipped(
                    "TextGrid structure",
                    "phones tier contains an interval with an empty label",
                )

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, paths, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in paths:
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
