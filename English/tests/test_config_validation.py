"""Unit tests for utils/config.py validation. No audio or datasets needed.

Run from English/ or Telugu/:
    python -m unittest tests.test_config_validation
"""
import copy
import os
import sys
import unittest

LANG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANG_DIR)

from utils.config import ConfigError, load_config, validate_config  # noqa: E402


def set_path(config, dotted, value):
    parts = dotted.split(".")
    target = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


class ConfigValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(os.path.join(LANG_DIR, "config.yaml"))
        self.language = self.config["preprocess"]["preprocessing"]["text"]["language"]

    def assert_invalid(self, mutate):
        config = copy.deepcopy(self.config)
        mutate(config)
        with self.assertRaises(ConfigError) as caught:
            validate_config(config, source="test-config")
        return str(caught.exception)

    def test_real_config_is_valid(self):
        validate_config(copy.deepcopy(self.config))

    def test_missing_section_is_reported(self):
        message = self.assert_invalid(lambda config: config.pop("model"))
        self.assertIn("missing section 'model'", message)

    def test_missing_required_field_is_reported(self):
        message = self.assert_invalid(
            lambda config: config["preprocess"]["path"].pop("manifest_train")
        )
        self.assertIn("manifest_train", message)

    def test_unknown_cleaner_is_reported(self):
        message = self.assert_invalid(
            lambda config: set_path(
                config,
                "preprocess.preprocessing.text.text_cleaners",
                ["not_a_cleaner"],
            )
        )
        self.assertIn("not a known cleaner", message)

    def test_cleaner_must_match_language(self):
        wrong = "english_cleaners" if self.language == "te" else "telugu_cleaners"
        message = self.assert_invalid(
            lambda config: set_path(
                config, "preprocess.preprocessing.text.text_cleaners", [wrong]
            )
        )
        self.assertIn("must use", message)

    def test_invalid_language_is_reported(self):
        message = self.assert_invalid(
            lambda config: set_path(
                config, "preprocess.preprocessing.text.language", "fr"
            )
        )
        self.assertIn("language", message)

    def test_zero_sampling_rate_is_reported(self):
        message = self.assert_invalid(
            lambda config: set_path(
                config, "preprocess.preprocessing.audio.sampling_rate", 0
            )
        )
        self.assertIn("sampling_rate", message)

    def test_boolean_is_not_accepted_as_integer(self):
        message = self.assert_invalid(
            lambda config: set_path(
                config, "preprocess.preprocessing.audio.sampling_rate", True
            )
        )
        self.assertIn("must be an integer", message)

    def test_mel_fmax_above_nyquist_is_reported(self):
        message = self.assert_invalid(
            lambda config: set_path(
                config, "preprocess.preprocessing.mel.mel_fmax", 20000
            )
        )
        self.assertIn("mel_fmax", message)

    def test_hop_longer_than_window_is_reported(self):
        message = self.assert_invalid(
            lambda config: set_path(
                config, "preprocess.preprocessing.stft.hop_length", 2048
            )
        )
        self.assertIn("hop_length", message)

    def test_filter_shorter_than_window_is_reported(self):
        message = self.assert_invalid(
            lambda config: set_path(
                config, "preprocess.preprocessing.stft.filter_length", 512
            )
        )
        self.assertIn("filter_length", message)

    def test_hidden_not_divisible_by_heads_is_reported(self):
        message = self.assert_invalid(
            lambda config: set_path(
                config, "model.transformer.encoder_hidden", 255
            )
        )
        self.assertIn("divisible", message)

    def test_dropout_of_one_is_reported(self):
        message = self.assert_invalid(
            lambda config: set_path(
                config, "model.transformer.encoder_dropout", 1.0
            )
        )
        self.assertIn("encoder_dropout", message)

    def test_conv_kernel_size_needs_two_values(self):
        message = self.assert_invalid(
            lambda config: set_path(
                config, "model.transformer.conv_kernel_size", [9]
            )
        )
        self.assertIn("conv_kernel_size", message)

    def test_n_bins_must_exceed_one(self):
        message = self.assert_invalid(
            lambda config: set_path(
                config, "model.variance_embedding.n_bins", 1
            )
        )
        self.assertIn("n_bins", message)

    def test_unknown_vocoder_is_reported(self):
        message = self.assert_invalid(
            lambda config: set_path(config, "model.vocoder.model", "Other")
        )
        self.assertIn("vocoder.model", message)

    def test_zero_batch_size_is_reported(self):
        message = self.assert_invalid(
            lambda config: set_path(config, "train.optimizer.batch_size", 0)
        )
        self.assertIn("batch_size", message)

    def test_zero_group_size_is_reported(self):
        message = self.assert_invalid(
            lambda config: set_path(config, "train.optimizer.group_size", 0)
        )
        self.assertIn("group_size", message)

    def test_negative_seed_is_reported(self):
        message = self.assert_invalid(
            lambda config: set_path(config, "train.seed", -1)
        )
        self.assertIn("seed", message)

    def test_save_step_larger_than_total_is_reported(self):
        message = self.assert_invalid(
            lambda config: set_path(config, "train.step.save_step", 1000000)
        )
        self.assertIn("save_step", message)

    def test_absolute_path_is_reported(self):
        for path in ("C:\\checkpoints", "/checkpoints", "~/checkpoints"):
            message = self.assert_invalid(
                lambda config, path=path: set_path(
                    config, "train.path.ckpt_path", path
                )
            )
            self.assertIn("relative path", message)

    def test_zero_duration_control_is_reported(self):
        message = self.assert_invalid(
            lambda config: set_path(config, "inference.duration_control", 0)
        )
        self.assertIn("duration_control", message)

    def test_restore_step_null_is_allowed(self):
        config = copy.deepcopy(self.config)
        config["inference"]["restore_step"] = None
        validate_config(config)

    def test_negative_restore_step_is_reported(self):
        message = self.assert_invalid(
            lambda config: set_path(config, "inference.restore_step", -1)
        )
        self.assertIn("restore_step", message)

    def test_english_requires_lexicon(self):
        if self.language != "en":
            self.skipTest("English-only rule")
        message = self.assert_invalid(
            lambda config: set_path(config, "preprocess.path.lexicon_path", "")
        )
        self.assertIn("lexicon_path", message)


if __name__ == "__main__":
    unittest.main()
