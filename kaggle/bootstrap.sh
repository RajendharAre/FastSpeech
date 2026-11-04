#!/usr/bin/env bash
# Bootstrap the Kaggle session for the FastSpeech 2 preprocessing pipeline.
#
# Steps:
#   1. Install Miniforge (skipped when conda is already available).
#   2. Create the "mfa" conda environment with Montreal Forced Aligner
#      (conda-forge) and Python 3.13.
#   3. Pip-install the pins that previously worked on a real Kaggle run:
#      NumPy 2.1.3, SciPy 1.16.3, librosa 0.10.2.post1, and pyworld built
#      with --no-build-isolation against that NumPy.
#   4. Unzip the pretrained HiFi-GAN generators for both languages.
#   5. Download the MFA english_us_arpa acoustic model and dictionary.
#
# Every step is idempotent and prints SKIP with its reason when the target
# already exists.
#
# STATUS: this script has NOT been run. The first real Kaggle run must
# record its output in CHANGES.md before any of it is treated as verified.
#
# Note: it does NOT install the full requirements.txt. Only the pins above
# are installed, on purpose; tgt/g2p_en and friends are a later decision.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-mfa}"
MFA_DATA_DIR="${MFA_DATA_DIR:-$HOME/Documents/MFA}"

step() { printf '\n==> %s\n' "$*"; }
skip() { printf 'SKIP: %s\n' "$*"; }

CONDA="$MINIFORGE_DIR/bin/conda"
if command -v conda >/dev/null 2>&1; then
    CONDA="$(command -v conda)"
elif [ ! -x "$CONDA" ]; then
    CONDA=""
fi

# --- 1. Miniforge -----------------------------------------------------------
step "1. Miniforge"
if [ -n "$CONDA" ]; then
    skip "conda already available at $CONDA"
else
    INSTALLER="/tmp/miniforge3-Linux-x86_64.sh"
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    echo "RUN: downloading $URL"
    wget -q -O "$INSTALLER" "$URL"
    echo "RUN: installing into $MINIFORGE_DIR"
    bash "$INSTALLER" -b -p "$MINIFORGE_DIR"
    rm -f "$INSTALLER"
    CONDA="$MINIFORGE_DIR/bin/conda"
    echo "DONE: Miniforge installed"
fi

ENV_PY="$MINIFORGE_DIR/envs/$ENV_NAME/bin/python"
# When conda came from the system rather than Miniforge, resolve the env
# python through that conda instead.
if [ ! -x "$ENV_PY" ] && [ -n "$CONDA" ]; then
    conda_envs_dir="$("$CONDA" info --base)/envs/$ENV_NAME"
    [ -x "$conda_envs_dir/bin/python" ] && ENV_PY="$conda_envs_dir/bin/python"
fi

# --- 2. MFA conda environment ----------------------------------------------
step "2. Conda environment '$ENV_NAME' (Montreal Forced Aligner, Python 3.13)"
if [ -x "$ENV_PY" ]; then
    skip "environment already exists at $(dirname "$(dirname "$ENV_PY")")"
else
    echo "RUN: conda create -n $ENV_NAME -c conda-forge python=3.13 montreal-forced-aligner"
    "$CONDA" create -y -n "$ENV_NAME" -c conda-forge python=3.13 montreal-forced-aligner
    echo "DONE: environment created"
fi
if [ ! -x "$ENV_PY" ]; then
    echo "ERROR: $ENV_PY not found after step 2; aborting before the pip pins." >&2
    exit 1
fi
"$ENV_PY" --version

# --- 3. Pip pins ------------------------------------------------------------
step "3. Pip pins (NumPy 2.1.3, SciPy 1.16.3, librosa 0.10.2.post1, pyworld)"
if "$ENV_PY" -c "import sys; assert sys.version_info[:2] == (3, 13), sys.version; \
import numpy, scipy, librosa, pyworld; \
assert numpy.__version__ == '2.1.3', numpy.__version__; \
assert scipy.__version__ == '1.16.3', scipy.__version__; \
assert librosa.__version__ == '0.10.2.post1', librosa.__version__" 2>/dev/null; then
    skip "all pins already importable at the required versions"
else
    # Build tools are intentionally NOT pinned here: the Cython version that
    # worked for the Python 3.12 build predates Python 3.13 support.
    echo "RUN: pip install Cython setuptools wheel"
    "$ENV_PY" -m pip install Cython setuptools wheel
    echo "RUN: pip install numpy==2.1.3 scipy==1.16.3 librosa==0.10.2.post1"
    "$ENV_PY" -m pip install "numpy==2.1.3" "scipy==1.16.3" "librosa==0.10.2.post1"
    echo "RUN: pip install --no-build-isolation pyworld==0.3.5"
    "$ENV_PY" -m pip install --no-build-isolation "pyworld==0.3.5"
    echo "DONE: pip pins installed"
fi

# --- 4. HiFi-GAN generators -------------------------------------------------
step "4. HiFi-GAN generators"
for lang in English Telugu; do
    hifi_dir="$ROOT/$lang/hifigan"
    zip_file="$hifi_dir/generator_universal.pth.tar.zip"
    extracted="$(find "$hifi_dir" -maxdepth 1 -name 'generator_universal*' ! -name '*.zip' 2>/dev/null | head -n 1 || true)"
    if [ -n "$extracted" ]; then
        skip "$lang generator already unzipped: $extracted"
    elif [ ! -f "$zip_file" ]; then
        skip "$lang zip not found at $zip_file (check the Kaggle dataset mount)"
    else
        echo "RUN: unzip -o $zip_file -d $hifi_dir"
        unzip -o "$zip_file" -d "$hifi_dir"
        echo "DONE: $lang generator unzipped"
    fi
done

# --- 5. MFA english_us_arpa models -----------------------------------------
# The cache paths under $MFA_DATA_DIR are the documented MFA defaults; if the
# path ever differs, the "already exists" guard below still keeps the script
# idempotent instead of failing on a re-download.
download_model() {
    kind="$1"
    name="$2"
    if out="$("$ENV_DIR/bin/mfa" model download "$kind" "$name" 2>&1)"; then
        printf '%s\n' "$out"
        echo "DONE: $kind model '$name' downloaded"
    else
        if printf '%s' "$out" | grep -qi "already exist"; then
            skip "MFA reports the $kind model '$name' already exists"
        else
            printf '%s\n' "$out" >&2
            echo "ERROR: $kind model '$name' download failed" >&2
            exit 1
        fi
    fi
}

step "5. MFA english_us_arpa acoustic model and dictionary"
if [ -f "$MFA_DATA_DIR/acoustic/english_us_arpa.zip" ] && \
   [ -f "$MFA_DATA_DIR/dictionary/english_us_arpa.dict" ]; then
    skip "both english_us_arpa models already present under $MFA_DATA_DIR"
else
    ENV_DIR="$(dirname "$ENV_PY")"
    download_model acoustic english_us_arpa
    download_model dictionary english_us_arpa
fi

step "Bootstrap finished"
echo "To use the environment:  conda activate $ENV_NAME"
echo "MFA data directory:      $MFA_DATA_DIR"
