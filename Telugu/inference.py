# NEW(both): inference entry point required by the assignment folder structure.
# Adapted from the reference synthesize.py (single-sentence mode only).
# Reads only the "final" field of docs/test_sentences.json and refuses to run
# while any final sentence is null (docs/RULES_FOR_AI.md, section 8).
import argparse
import json
import os
import re
from string import punctuation

import matplotlib
import numpy as np
import torch
from matplotlib import font_manager, pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import FT2Font
from scipy.io import wavfile

from model import FastSpeech2
from text import text_to_sequence
from text.symbols import symbols as SYMBOLS
from utils.config import load_config
from utils.model import get_vocoder, vocoder_infer
from utils.train_utils import load_checkpoint
from utils.tools import to_device

matplotlib.use("Agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LANGUAGE_SECTION = "Telugu"

AUDIO_DIR_NAME = "test_generated_audios"
VARIANCE_DIR_NAME = "variance_adaptor_plots"


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    """Reference English frontend: lexicon lookup, then g2p_en for unknown words."""
    from g2p_en import G2p

    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return sequence


def preprocess_text(text, preprocess_config):
    language = preprocess_config["preprocessing"]["text"]["language"]
    if language == "en":
        return preprocess_english(text, preprocess_config)
    if language == "te":
        # MODIFIED(Telugu): character-level frontend over a frozen Unicode set.
        # Unsupported characters raise instead of being dropped silently.
        print("Raw Text Sequence: {}".format(text))
        return np.array(
            text_to_sequence(
                text,
                preprocess_config["preprocessing"]["text"]["text_cleaners"],
            )
        )
    raise NotImplementedError(
        "No frontend for language '{}'. Supported languages: en, te.".format(language)
    )


def load_test_sentences(path, language):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if language not in data:
        raise KeyError("{} has no '{}' section".format(path, language))
    entries = data[language]
    missing = [e["id"] for e in entries if not e.get("final")]
    if missing:
        raise SystemExit(
            "Refusing to run: docs/test_sentences.json still has null 'final' "
            "text for {}. Fill them from the rendered PDF first.".format(
                ", ".join(missing)
            )
        )
    return entries


def symbols_from_ids(sequence):
    """Map symbol IDs back to the symbols the model was conditioned on."""
    return [SYMBOLS[int(symbol_id)] for symbol_id in sequence]


def _font_covers(font_path, characters):
    try:
        font = FT2Font(font_path)
    except Exception:
        return False
    return all(font.get_char_index(ord(character)) for character in characters)


_FALLBACK_FONT_FAMILIES = (
    "Noto Sans Telugu",
    "Noto Serif Telugu",
    "Telugu Sangam MN",
    "Lohit Telugu",
    "Nirmala UI",
    "Gautami",
)


def symbol_labels(symbols):
    """Return axis labels plus the FontProperties needed to render them.

    matplotlib's default font has no Telugu glyphs. If no installed font covers
    the symbols the labels fall back to code points, so a missing font degrades
    the tick labels rather than producing empty boxes in a report figure.
    """
    characters = {character for symbol in symbols for character in symbol}
    if _font_covers(font_manager.findfont(FontProperties()), characters):
        return list(symbols), None

    for family in _FALLBACK_FONT_FAMILIES:
        try:
            path = font_manager.findfont(
                FontProperties(family=family), fallback_to_default=False
            )
        except Exception:
            continue
        if _font_covers(path, characters):
            font_manager.fontManager.addfont(path)
            name = FontProperties(fname=path).get_name()
            return list(symbols), FontProperties(family=name, size=7)

    print(
        "WARNING: no installed font covers the model symbols; plotting code "
        "points instead. Install a Telugu font (for example Noto Sans Telugu) "
        "for readable axis labels."
    )
    return [
        " ".join("U+{:04X}".format(ord(character)) for character in symbol)
        for symbol in symbols
    ], None


def denormalize(values, stat_block):
    """Undo the standardisation stored in stats.json ([min, max, mean, std])."""
    mean, std = float(stat_block[2]), float(stat_block[3])
    return np.asarray(values, dtype=np.float64) * std + mean


def load_feature_stats(preprocess_config):
    stats_path = os.path.join(
        preprocess_config["path"]["preprocessed_path"], "stats.json"
    )
    if not os.path.isfile(stats_path):
        raise SystemExit(
            "Missing {}. The variance-adaptor export denormalises pitch and "
            "energy with the statistics written by preprocess.py.".format(stats_path)
        )
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_series(values, labels, out_path, title, ylabel, fontproperties=None, kind="bar"):
    fig, ax = plt.subplots(figsize=(max(6.0, 0.3 * len(values)), 4.0))
    x = np.arange(len(values))
    if kind == "bar":
        ax.bar(x, values, color="steelblue")
    else:
        ax.plot(x, values, color="steelblue", marker="o")
    stride = max(1, len(values) // 40)
    ax.set_xticks(x[::stride])
    ax.set_xticklabels(
        labels[::stride], rotation=90, fontproperties=fontproperties, fontsize=7
    )
    ax.set_xlabel("symbol index")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def export_variance_outputs(result_path, entry, symbols, arrays, meta):
    """Write duration/pitch/energy plots and the predicted arrays for one sentence.

    The plots go to ``variance_adaptor_plots/<kind>/<ID>.png`` as the deliverable
    requires; the arrays go to ``variance_adaptor_plots/<ID>.json`` so the report
    can quote numbers without re-running inference.
    """
    for name, values in arrays.items():
        if len(values) != len(symbols):
            raise ValueError(
                "{} has {} value(s) but the sequence has {} symbol(s)".format(
                    name, len(values), len(symbols)
                )
            )

    root = os.path.join(result_path, VARIANCE_DIR_NAME)
    labels, fontproperties = symbol_labels(symbols)
    paths = []
    for name, values, ylabel, kind in (
        ("duration", arrays["duration_frames"], "predicted duration (frames)", "bar"),
        ("pitch", arrays["pitch_hz"], "predicted F0 (Hz)", "line"),
        ("energy", arrays["energy"], "predicted energy", "line"),
    ):
        out_dir = os.path.join(root, name)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "{}.png".format(entry["id"]))
        plot_series(
            values,
            labels,
            path,
            "{} / predicted {} ({} symbols)".format(entry["id"], name, len(symbols)),
            ylabel,
            fontproperties,
            kind,
        )
        paths.append(path)

    payload = {"id": entry["id"], "text": entry["final"], "symbols": list(symbols)}
    payload.update({name: [float(v) for v in values] for name, values in arrays.items()})
    payload.update(meta)
    json_path = os.path.join(root, "{}.json".format(entry["id"]))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    paths.append(json_path)
    return paths


def check_phoneme_level(preprocess_config):
    levels = {
        name: preprocess_config["preprocessing"][name]["feature"]
        for name in ("pitch", "energy")
    }
    if any(value != "phoneme_level" for value in levels.values()):
        raise SystemExit(
            "The variance-adaptor export reads pitch and energy at the symbol "
            "level, but the config uses {}; run preprocess.py with "
            "phoneme_level features.".format(json.dumps(levels))
        )


def synthesize(
    model, vocoder, configs, entries, controls, speaker_id, result_path, stats, checkpoint_step
):
    preprocess_config, model_config = configs
    pitch_control, energy_control, duration_control = controls
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]

    audio_dir = os.path.join(result_path, AUDIO_DIR_NAME)
    os.makedirs(audio_dir, exist_ok=True)
    model.eval()
    for entry in entries:
        text = entry["final"]
        sequence = preprocess_text(text, preprocess_config)
        batch = (
            [entry["id"]],
            [text],
            np.array([speaker_id]),
            np.array([sequence]),
            np.array([len(sequence)]),
            len(sequence),
        )
        batch = to_device(batch, device)
        with torch.no_grad():
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
            )
        src_len = output[8][0].item()
        mel_len = output[9][0].item()
        mel = output[1][0, :mel_len].transpose(0, 1)
        wav = vocoder_infer(
            mel.unsqueeze(0), vocoder, model_config, preprocess_config
        )[0]
        wav_path = os.path.join(audio_dir, "{}.wav".format(entry["id"]))
        wavfile.write(wav_path, sampling_rate, wav)
        print("Wrote {} ({} frames)".format(wav_path, mel_len))

        if src_len != len(sequence):
            raise RuntimeError(
                "model reports {} symbols but the frontend produced {}".format(
                    src_len, len(sequence)
                )
            )
        durations = np.rint(output[5][0, :src_len].detach().cpu().numpy()).astype(int)
        pitch_normalized = output[2][0, :src_len].detach().cpu().numpy().reshape(-1)
        energy_normalized = output[3][0, :src_len].detach().cpu().numpy().reshape(-1)
        arrays = {
            "duration_frames": durations.astype(float),
            "duration_seconds": durations.astype(float) * hop_length / sampling_rate,
            "pitch_hz": denormalize(pitch_normalized, stats["pitch"]),
            "energy": denormalize(energy_normalized, stats["energy"]),
            "pitch_normalized": pitch_normalized,
            "energy_normalized": energy_normalized,
        }
        meta = {
            "checkpoint_step": int(checkpoint_step),
            "language": preprocess_config["preprocessing"]["text"]["language"],
            "speaker_id": int(speaker_id),
            "controls": {
                "pitch_control": float(pitch_control),
                "energy_control": float(energy_control),
                "duration_control": float(duration_control),
            },
            "sampling_rate": int(sampling_rate),
            "hop_length": int(hop_length),
            "mel_frames": int(mel_len),
            "stats": stats,
        }
        for path in export_variance_outputs(result_path, entry, symbols_from_ids(sequence), arrays, meta):
            print("  wrote {}".format(path))


def main(args):
    config = load_config(args.config)
    preprocess_config = config["preprocess"]
    model_config = config["model"]
    inference_config = config["inference"]

    entries = load_test_sentences(
        inference_config["test_sentences"], LANGUAGE_SECTION
    )
    check_phoneme_level(preprocess_config)

    step = args.restore_step or inference_config["restore_step"]
    if not step:
        raise SystemExit(
            "No checkpoint step given. Pass --restore_step or set "
            "inference.restore_step in config.yaml."
        )

    ckpt_path = os.path.join(
        config["train"]["path"]["ckpt_path"], "{}.pth.tar".format(step)
    )
    if not os.path.exists(ckpt_path):
        raise SystemExit("Checkpoint not found: {}".format(ckpt_path))

    vocoder_ckpt = os.path.join("hifigan", "generator_universal.pth.tar")
    if not os.path.exists(vocoder_ckpt):
        raise SystemExit(
            "Missing {}. Extract it first: "
            "unzip hifigan/generator_universal.pth.tar.zip -d hifigan".format(
                vocoder_ckpt
            )
        )

    model = FastSpeech2(preprocess_config, model_config).to(device)
    ckpt = load_checkpoint(ckpt_path, device)
    model.load_state_dict(ckpt["model"])

    vocoder = get_vocoder(model_config, device)

    controls = (
        inference_config["pitch_control"],
        inference_config["energy_control"],
        inference_config["duration_control"],
    )
    synthesize(
        model,
        vocoder,
        (preprocess_config, model_config),
        entries,
        controls,
        inference_config["speaker_id"],
        inference_config["result_path"],
        load_feature_stats(preprocess_config),
        step,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--restore_step", type=int, default=None)
    args = parser.parse_args()
    main(args)
