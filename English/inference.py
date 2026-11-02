# NEW(both): inference entry point required by the assignment folder structure.
# Adapted from the reference synthesize.py (single-sentence mode only).
# Reads only the "final" field of docs/test_sentences.json and refuses to run
# while any final sentence is null (docs/RULES_FOR_AI.md, section 8).
import argparse
import json
import os
import re
from string import punctuation

import numpy as np
import torch
from scipy.io import wavfile

from model import FastSpeech2
from text import text_to_sequence
from utils.config import load_config
from utils.model import get_vocoder, vocoder_infer
from utils.tools import to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LANGUAGE_SECTION = "English"


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
    # MODIFIED(Telugu): the Telugu frontend (symbol set + number-to-words) is a
    # GATE 1 decision and is built in Phase 3. Failing loudly is intentional.
    raise NotImplementedError(
        "No frontend for language '{}' yet. See docs/PLAN_PHASES.md Phase 3.".format(
            language
        )
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


def synthesize(
    model, vocoder, configs, entries, controls, speaker_id, out_dir
):
    preprocess_config, model_config = configs
    pitch_control, energy_control, duration_control = controls
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    os.makedirs(out_dir, exist_ok=True)
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
        mel_len = output[9][0].item()
        mel = output[1][0, :mel_len].transpose(0, 1)
        wav = vocoder_infer(
            mel.unsqueeze(0), vocoder, model_config, preprocess_config
        )[0]
        wav_path = os.path.join(out_dir, "{}.wav".format(entry["id"]))
        wavfile.write(wav_path, sampling_rate, wav)
        print("Wrote {} ({} frames)".format(wav_path, mel_len))


def main(args):
    config = load_config(args.config)
    preprocess_config = config["preprocess"]
    model_config = config["model"]
    inference_config = config["inference"]

    entries = load_test_sentences(
        inference_config["test_sentences"], LANGUAGE_SECTION
    )

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
    ckpt = torch.load(ckpt_path, map_location=device)
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
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--restore_step", type=int, default=None)
    args = parser.parse_args()
    main(args)
