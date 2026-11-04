import argparse

from preprocessor.preprocessor import Preprocessor

# MODIFIED(both): read the single per-language config.yaml and hand the
# "preprocess" section to the reference Preprocessor unchanged.
from utils.config import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="config.yaml",
        help="path to the language config.yaml",
    )
    parser.add_argument(
        "--allow-dropped",
        action="store_true",
        help=(
            "continue after clips are dropped; by default any dropped clip "
            "stops the run before normalisation"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    preprocessor = Preprocessor(
        config["preprocess"],
        allow_dropped=args.allow_dropped,
        max_seq_len=config["model"]["max_seq_len"],
    )
    preprocessor.build_from_path()
