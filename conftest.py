"""Make the repository root importable under a plain ``pytest`` invocation.

``python -m pytest`` puts the current directory on ``sys.path``; the pytest
console script does not, so ``import scripts.validate_alignments`` failed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
