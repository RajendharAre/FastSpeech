# NEW(both): the assignment requires one config.yaml per language with the
# sections preprocess, model, train and inference. The reference implementation
# used three separate YAML files; this loader keeps a single file and hands the
# existing code the same dictionaries it already expects.
#
# MODIFIED(both): load_config now validates required fields and value ranges
# before preprocessing or training starts, so a bad config fails immediately
# instead of failing in the middle of a long job.
import math
import os
import re

import yaml

REQUIRED_SECTIONS = ("preprocess", "model", "train", "inference")

_KNOWN_CLEANERS = {
    "basic_cleaners",
    "english_cleaners",
    "transliteration_cleaners",
    "telugu_cleaners",
}
_LANGUAGES = {"en", "te"}
_FEATURES = {"phoneme_level", "frame_level"}
_QUANTIZATIONS = {"linear", "log"}
_VOCODER_MODELS = {"HiFi-GAN", "MelGAN"}
_VOCODER_SPEAKERS = {"universal", "LJSpeech"}
_MISSING = object()


class ConfigError(ValueError):
    """Raised when config.yaml is missing fields or has invalid values."""


def _get(errors, mapping, key, prefix):
    field = "{}.{}".format(prefix, key) if prefix else key
    if not isinstance(mapping, dict):
        errors.append("{} must be a mapping".format(prefix or key))
        return _MISSING
    if key not in mapping:
        errors.append("missing required field '{}'".format(field))
        return _MISSING
    return mapping[key]


def _is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_str(errors, value, field, allow_empty=False):
    if value is _MISSING:
        return
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        errors.append(
            "{} must be a{} string".format(field, "" if allow_empty else " non-empty")
        )


def _check_bool(errors, value, field):
    if value is not _MISSING and not isinstance(value, bool):
        errors.append("{} must be true or false".format(field))


def _check_int(errors, value, field, minimum=None):
    if value is _MISSING:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append("{} must be an integer".format(field))
        return
    if minimum is not None and value < minimum:
        errors.append("{} must be >= {}".format(field, minimum))


def _check_number(
    errors, value, field, minimum=None, maximum=None, inclusive_min=True, inclusive_max=True
):
    if value is _MISSING:
        return
    if not _is_number(value):
        errors.append("{} must be a finite number".format(field))
        return
    if minimum is not None:
        if value < minimum or (not inclusive_min and value == minimum):
            errors.append(
                "{} must be {} {}".format(
                    field, ">=" if inclusive_min else ">", minimum
                )
            )
    if maximum is not None:
        if value > maximum or (not inclusive_max and value == maximum):
            errors.append(
                "{} must be {} {}".format(
                    field, "<=" if inclusive_max else "<", maximum
                )
            )


def _check_choice(errors, value, field, choices):
    if value is _MISSING:
        return
    if not isinstance(value, str) or value not in choices:
        errors.append(
            "{} must be one of: {}".format(field, ", ".join(sorted(choices)))
        )


def _check_int_list(errors, value, field, minimum_length=0, minimum_value=None):
    if value is _MISSING:
        return
    if not isinstance(value, list):
        errors.append("{} must be a list of integers".format(field))
        return
    if len(value) < minimum_length:
        errors.append(
            "{} must contain at least {} item(s)".format(field, minimum_length)
        )
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            errors.append("{}[{}] must be an integer".format(field, index))
        elif minimum_value is not None and item < minimum_value:
            errors.append("{}[{}] must be >= {}".format(field, index, minimum_value))


def _check_number_list(errors, value, field, length, minimum, maximum):
    if value is _MISSING:
        return
    if not isinstance(value, list) or len(value) != length:
        errors.append("{} must be a list of {} numbers".format(field, length))
        return
    for index, item in enumerate(value):
        _check_number(
            errors,
            item,
            "{}[{}]".format(field, index),
            minimum=minimum,
            maximum=maximum,
        )


def _check_relative_path(errors, value, field, allow_empty=False):
    _check_str(errors, value, field, allow_empty=allow_empty)
    if value is _MISSING or not isinstance(value, str) or not value.strip():
        return
    absolute_like = (
        os.path.isabs(value)
        or value.startswith(("~", "/", "\\"))
        or bool(re.match(r"^[A-Za-z]:[\\/]", value))
    )
    if absolute_like:
        errors.append(
            "{} must be a relative path (run scripts from the language folder), "
            "got {!r}".format(field, value)
        )


def _validate_preprocess(errors, preprocess):
    _check_str(errors, _get(errors, preprocess, "dataset", "preprocess"), "preprocess.dataset")

    path = _get(errors, preprocess, "path", "preprocess")
    path_values = {}
    for key in (
        "corpus_path",
        "raw_path",
        "preprocessed_path",
        "manifest_train",
        "manifest_val",
        "lexicon_path",
    ):
        field = "preprocess.path.{}".format(key)
        value = _get(errors, path, key, "preprocess.path")
        path_values[key] = value
        _check_relative_path(errors, value, field, allow_empty=(key == "lexicon_path"))

    preprocessing = _get(errors, preprocess, "preprocessing", "preprocess")

    text = _get(errors, preprocessing, "text", "preprocess.preprocessing")
    language = _get(errors, text, "language", "preprocess.preprocessing.text")
    _check_choice(errors, language, "preprocess.preprocessing.text.language", _LANGUAGES)

    cleaners = _get(
        errors, text, "text_cleaners", "preprocess.preprocessing.text"
    )
    valid_cleaners = []
    if cleaners is not _MISSING:
        if not isinstance(cleaners, list) or not cleaners:
            errors.append(
                "preprocess.preprocessing.text.text_cleaners must be a non-empty list"
            )
        else:
            for index, name in enumerate(cleaners):
                if not isinstance(name, str) or name not in _KNOWN_CLEANERS:
                    errors.append(
                        "preprocess.preprocessing.text.text_cleaners[{}] is not a "
                        "known cleaner (known: {})".format(
                            index, ", ".join(sorted(_KNOWN_CLEANERS))
                        )
                    )
                else:
                    valid_cleaners.append(name)

    if language == "te" and valid_cleaners and "telugu_cleaners" not in valid_cleaners:
        errors.append(
            "Telugu config (language 'te') must use 'telugu_cleaners'"
        )
    if language == "en" and valid_cleaners and "english_cleaners" not in valid_cleaners:
        errors.append(
            "English config (language 'en') must use 'english_cleaners'"
        )
    if (
        language == "en"
        and isinstance(path_values.get("lexicon_path"), str)
        and not path_values["lexicon_path"].strip()
    ):
        errors.append("English config requires a non-empty preprocess.path.lexicon_path")

    audio = _get(errors, preprocessing, "audio", "preprocess.preprocessing")
    sampling_rate = _get(
        errors, audio, "sampling_rate", "preprocess.preprocessing.audio"
    )
    _check_int(
        errors,
        sampling_rate,
        "preprocess.preprocessing.audio.sampling_rate",
        minimum=1,
    )
    _check_number(
        errors,
        _get(errors, audio, "max_wav_value", "preprocess.preprocessing.audio"),
        "preprocess.preprocessing.audio.max_wav_value",
        minimum=0,
        inclusive_min=False,
    )

    stft = _get(errors, preprocessing, "stft", "preprocess.preprocessing")
    filter_length = _get(
        errors, stft, "filter_length", "preprocess.preprocessing.stft"
    )
    hop_length = _get(errors, stft, "hop_length", "preprocess.preprocessing.stft")
    win_length = _get(errors, stft, "win_length", "preprocess.preprocessing.stft")
    for name, value in (
        ("filter_length", filter_length),
        ("hop_length", hop_length),
        ("win_length", win_length),
    ):
        _check_int(
            errors,
            value,
            "preprocess.preprocessing.stft.{}".format(name),
            minimum=1,
        )
    if all(isinstance(v, int) and not isinstance(v, bool) for v in (filter_length, hop_length, win_length)):
        if filter_length < win_length:
            errors.append(
                "preprocess.preprocessing.stft.filter_length must be >= win_length"
            )
        if hop_length > win_length:
            errors.append(
                "preprocess.preprocessing.stft.hop_length must be <= win_length"
            )

    mel = _get(errors, preprocessing, "mel", "preprocess.preprocessing")
    _check_int(
        errors,
        _get(errors, mel, "n_mel_channels", "preprocess.preprocessing.mel"),
        "preprocess.preprocessing.mel.n_mel_channels",
        minimum=1,
    )
    mel_fmin = _get(errors, mel, "mel_fmin", "preprocess.preprocessing.mel")
    mel_fmax = _get(errors, mel, "mel_fmax", "preprocess.preprocessing.mel")
    _check_number(
        errors, mel_fmin, "preprocess.preprocessing.mel.mel_fmin", minimum=0
    )
    _check_number(
        errors, mel_fmax, "preprocess.preprocessing.mel.mel_fmax", minimum=0
    )
    if _is_number(mel_fmin) and _is_number(mel_fmax):
        if mel_fmin >= mel_fmax:
            errors.append(
                "preprocess.preprocessing.mel.mel_fmin must be < mel_fmax"
            )
        if _is_number(sampling_rate) and mel_fmax > sampling_rate / 2:
            errors.append(
                "preprocess.preprocessing.mel.mel_fmax must be <= sampling_rate / 2"
            )

    for name in ("pitch", "energy"):
        block = _get(errors, preprocessing, name, "preprocess.preprocessing")
        prefix = "preprocess.preprocessing.{}".format(name)
        _check_choice(
            errors, _get(errors, block, "feature", prefix), prefix + ".feature", _FEATURES
        )
        _check_bool(
            errors,
            _get(errors, block, "normalization", prefix),
            prefix + ".normalization",
        )


def _validate_model(errors, model):
    transformer = _get(errors, model, "transformer", "model")
    prefix = "model.transformer"
    positive_ints = {}
    for key in (
        "encoder_layer",
        "encoder_head",
        "encoder_hidden",
        "decoder_layer",
        "decoder_head",
        "decoder_hidden",
        "conv_filter_size",
    ):
        value = _get(errors, transformer, key, prefix)
        positive_ints[key] = value
        _check_int(errors, value, "{}.{}".format(prefix, key), minimum=1)

    _check_int_list(
        errors,
        _get(errors, transformer, "conv_kernel_size", prefix),
        prefix + ".conv_kernel_size",
        minimum_length=2,
        minimum_value=1,
    )
    for key in ("encoder_dropout", "decoder_dropout"):
        _check_number(
            errors,
            _get(errors, transformer, key, prefix),
            "{}.{}".format(prefix, key),
            minimum=0,
            maximum=1,
            inclusive_max=False,
        )

    encoder_hidden = positive_ints.get("encoder_hidden")
    encoder_head = positive_ints.get("encoder_head")
    decoder_hidden = positive_ints.get("decoder_hidden")
    decoder_head = positive_ints.get("decoder_head")
    if (
        isinstance(encoder_hidden, int)
        and not isinstance(encoder_hidden, bool)
        and isinstance(encoder_head, int)
        and not isinstance(encoder_head, bool)
        and encoder_hidden % encoder_head != 0
    ):
        errors.append("model.transformer.encoder_hidden must be divisible by encoder_head")
    if (
        isinstance(decoder_hidden, int)
        and not isinstance(decoder_hidden, bool)
        and isinstance(decoder_head, int)
        and not isinstance(decoder_head, bool)
        and decoder_hidden % decoder_head != 0
    ):
        errors.append("model.transformer.decoder_hidden must be divisible by decoder_head")

    predictor = _get(errors, model, "variance_predictor", "model")
    prefix = "model.variance_predictor"
    _check_int(
        errors, _get(errors, predictor, "filter_size", prefix), prefix + ".filter_size", minimum=1
    )
    _check_int(
        errors, _get(errors, predictor, "kernel_size", prefix), prefix + ".kernel_size", minimum=1
    )
    _check_number(
        errors,
        _get(errors, predictor, "dropout", prefix),
        prefix + ".dropout",
        minimum=0,
        maximum=1,
        inclusive_max=False,
    )

    embedding = _get(errors, model, "variance_embedding", "model")
    prefix = "model.variance_embedding"
    for key in ("pitch_quantization", "energy_quantization"):
        _check_choice(
            errors, _get(errors, embedding, key, prefix), "{}.{}".format(prefix, key), _QUANTIZATIONS
        )
    _check_int(
        errors,
        _get(errors, embedding, "n_bins", prefix),
        prefix + ".n_bins",
        minimum=2,
    )

    _check_bool(errors, _get(errors, model, "multi_speaker", "model"), "model.multi_speaker")
    _check_int(
        errors,
        _get(errors, model, "max_seq_len", "model"),
        "model.max_seq_len",
        minimum=1,
    )

    vocoder = _get(errors, model, "vocoder", "model")
    _check_choice(
        errors,
        _get(errors, vocoder, "model", "model.vocoder"),
        "model.vocoder.model",
        _VOCODER_MODELS,
    )
    _check_choice(
        errors,
        _get(errors, vocoder, "speaker", "model.vocoder"),
        "model.vocoder.speaker",
        _VOCODER_SPEAKERS,
    )


def _validate_train(errors, train):
    path = _get(errors, train, "path", "train")
    for key in ("ckpt_path", "log_path", "result_path"):
        _check_relative_path(
            errors,
            _get(errors, path, key, "train.path"),
            "train.path.{}".format(key),
        )

    _check_int(errors, _get(errors, train, "seed", "train"), "train.seed", minimum=0)

    optimizer = _get(errors, train, "optimizer", "train")
    prefix = "train.optimizer"
    _check_int(
        errors,
        _get(errors, optimizer, "batch_size", prefix),
        prefix + ".batch_size",
        minimum=1,
    )
    _check_int(
        errors,
        _get(errors, optimizer, "group_size", prefix),
        prefix + ".group_size",
        minimum=1,
    )
    _check_number_list(
        errors,
        _get(errors, optimizer, "betas", prefix),
        prefix + ".betas",
        length=2,
        minimum=0,
        maximum=1,
    )
    _check_number(
        errors,
        _get(errors, optimizer, "eps", prefix),
        prefix + ".eps",
        minimum=0,
        inclusive_min=False,
    )
    _check_number(
        errors,
        _get(errors, optimizer, "weight_decay", prefix),
        prefix + ".weight_decay",
        minimum=0,
    )
    _check_number(
        errors,
        _get(errors, optimizer, "grad_clip_thresh", prefix),
        prefix + ".grad_clip_thresh",
        minimum=0,
        inclusive_min=False,
    )
    _check_int(
        errors,
        _get(errors, optimizer, "grad_acc_step", prefix),
        prefix + ".grad_acc_step",
        minimum=1,
    )
    _check_int(
        errors,
        _get(errors, optimizer, "warm_up_step", prefix),
        prefix + ".warm_up_step",
        minimum=1,
    )
    _check_int_list(
        errors,
        _get(errors, optimizer, "anneal_steps", prefix),
        prefix + ".anneal_steps",
        minimum_value=0,
    )
    _check_number(
        errors,
        _get(errors, optimizer, "anneal_rate", prefix),
        prefix + ".anneal_rate",
        minimum=0,
        maximum=1,
        inclusive_min=False,
    )

    step = _get(errors, train, "step", "train")
    prefix = "train.step"
    step_values = {}
    for key in ("total_step", "log_step", "synth_step", "val_step", "save_step"):
        value = _get(errors, step, key, prefix)
        step_values[key] = value
        _check_int(errors, value, "{}.{}".format(prefix, key), minimum=1)
    total_step = step_values.get("total_step")
    if isinstance(total_step, int) and not isinstance(total_step, bool):
        for key in ("log_step", "synth_step", "val_step", "save_step"):
            value = step_values.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > total_step:
                errors.append(
                    "train.step.{} must be <= train.step.total_step".format(key)
                )


def _validate_inference(errors, inference):
    restore_step = _get(errors, inference, "restore_step", "inference")
    if restore_step is not _MISSING and restore_step is not None:
        _check_int(
            errors,
            restore_step,
            "inference.restore_step",
            minimum=1,
        )
    _check_int(
        errors,
        _get(errors, inference, "speaker_id", "inference"),
        "inference.speaker_id",
        minimum=0,
    )
    for key in ("pitch_control", "energy_control", "duration_control"):
        _check_number(
            errors,
            _get(errors, inference, key, "inference"),
            "inference.{}".format(key),
            minimum=0,
            inclusive_min=False,
        )
    _check_relative_path(
        errors,
        _get(errors, inference, "test_sentences", "inference"),
        "inference.test_sentences",
    )
    _check_relative_path(
        errors,
        _get(errors, inference, "result_path", "inference"),
        "inference.result_path",
    )


def validate_config(config, source="config"):
    """Validate a full config mapping; raise ConfigError listing every problem."""
    if not isinstance(config, dict):
        raise ConfigError("{} must contain a YAML mapping".format(source))

    errors = []
    for section in REQUIRED_SECTIONS:
        if section not in config:
            errors.append("missing section '{}'".format(section))
        elif not isinstance(config[section], dict):
            errors.append("section '{}' must be a mapping".format(section))
    if errors:
        raise ConfigError(
            "Invalid config {}:\n  - {}".format(source, "\n  - ".join(errors))
        )

    _validate_preprocess(errors, config["preprocess"])
    _validate_model(errors, config["model"])
    _validate_train(errors, config["train"])
    _validate_inference(errors, config["inference"])

    if errors:
        raise ConfigError(
            "Invalid config {}:\n  - {}".format(source, "\n  - ".join(errors))
        )
    return config


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return validate_config(config, source=path)


def split_config(config):
    """Return the (preprocess, model, train) tuple used by the reference code."""
    validate_config(config)
    return config["preprocess"], config["model"], config["train"]
