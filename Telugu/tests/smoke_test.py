"""CPU smoke test: build FastSpeech2 from the language config and run one
forward pass on a tiny random batch. No training, no checkpoint is loaded.

Run from the language folder (English/ or Telugu/):
    python tests/smoke_test.py
"""
import copy
import json
import os
import shutil
import sys
import tempfile
import time

import torch

LANG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANG_DIR)

from model import FastSpeech2  # noqa: E402
from utils.config import load_config, split_config  # noqa: E402


def main():
    torch.manual_seed(1234)
    config = load_config(os.path.join(LANG_DIR, "config.yaml"))
    preprocess_config, model_config, _ = split_config(config)
    n_mels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]

    tmp = tempfile.mkdtemp(prefix="fs2_smoke_")
    try:
        # VarianceAdaptor reads pitch/energy ranges at construction time.
        with open(os.path.join(tmp, "stats.json"), "w") as f:
            json.dump(
                {"pitch": [0.0, 1.0, 0.0, 1.0], "energy": [0.0, 1.0, 0.0, 1.0]}, f
            )
        if model_config["multi_speaker"]:
            with open(os.path.join(tmp, "speakers.json"), "w") as f:
                json.dump({"smoke_speaker": 0}, f)

        preprocess_config = copy.deepcopy(preprocess_config)
        preprocess_config["path"]["preprocessed_path"] = tmp
        model = FastSpeech2(preprocess_config, model_config)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        print("config:", os.path.join(LANG_DIR, "config.yaml"))
        print("multi_speaker:", model_config["multi_speaker"])
        print("parameters:", "{:,}".format(n_params))

        # Tiny random batch: B=2, T=12 symbols, 36 and 27 output frames.
        batch_size, max_src_len = 2, 12
        speakers = torch.zeros(batch_size, dtype=torch.long)
        texts = torch.randint(1, 20, (batch_size, max_src_len), dtype=torch.long)
        src_lens = torch.tensor([12, 9])
        durations = torch.zeros(batch_size, max_src_len, dtype=torch.long)
        durations[0] = 3
        durations[1, :9] = 3
        mel_lens = durations.sum(dim=1)
        max_mel_len = int(mel_lens.max())
        mels = torch.randn(batch_size, max_mel_len, n_mels)
        pitches = torch.randn(batch_size, max_src_len)
        energies = torch.randn(batch_size, max_src_len)

        with torch.no_grad():
            start = time.perf_counter()
            output = model(
                speakers,
                texts,
                src_lens,
                max_src_len,
                mels,
                mel_lens,
                max_mel_len,
                pitches,
                energies,
                durations,
            )
            elapsed = time.perf_counter() - start

        (
            mel_out,
            postnet_out,
            pitch_out,
            energy_out,
            log_duration_out,
            duration_out,
            _src_masks,
            _mel_masks,
            _src_lens,
            mel_lens_out,
        ) = output

        checks = {
            "mel": ((mel_out.shape, (batch_size, max_mel_len, n_mels))),
            "mel_postnet": ((postnet_out.shape, (batch_size, max_mel_len, n_mels))),
            "pitch": ((pitch_out.shape, (batch_size, max_src_len))),
            "energy": ((energy_out.shape, (batch_size, max_src_len))),
            "log_duration": ((log_duration_out.shape, (batch_size, max_src_len))),
            "duration": ((duration_out.shape, (batch_size, max_src_len))),
        }
        for name, (got, want) in checks.items():
            assert got == want, "{} shape {} != {}".format(name, got, want)
        assert torch.equal(mel_lens_out, mel_lens), "regulator lengths differ"
        assert torch.isfinite(mel_out).all(), "non-finite values in mel output"

        print("forward pass: OK in {:.3f} s".format(elapsed))
        for name, (got, _want) in checks.items():
            print("  {:<13} {}".format(name, tuple(got)))
        print("  mel_lens_out  {}".format(mel_lens_out.tolist()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
