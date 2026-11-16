"""rashomon-zoo: many equally good black boxes per task and their local explanations."""
import os as _os
import sys as _sys

# macOS: torch and xgboost load different libomp copies (torch bundles its own, xgboost links Homebrew's).
# With both active, xgboost's fit segfaults (torch imported first) or torch deadlocks (xgboost imported first).
# One OpenMP thread avoids both (checked: torch 2.15, xgboost 3.4.1, Python 3.13, arm64). Parallelise across
# models with processes instead. Set OMP_NUM_THREADS yourself before importing to override.
if _sys.platform == "darwin":
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
