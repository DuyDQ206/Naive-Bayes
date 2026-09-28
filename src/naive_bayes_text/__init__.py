"""Naive Bayes for text classification - CS313 group project.
"""
from __future__ import annotations

__version__ = "1.0.0"

from .config import (  # noqa: F401
    ALPHAS,
    BENCH_VEC_PARAMS,
    SEED,
    TASKS,
    TEST_SIZE,
    THRESHOLD_GRID,
    VAL_SIZE,
    VEC_PARAMS,
)
from .data import clean_text, load_all, load_task, prepare_all  # noqa: F401
from .modeling import (  # noqa: F401
    evaluate,
    make_pipeline,
    model_size_kb,
    pick_threshold,
    tune_alpha,
)
from .pipeline import train_all  # noqa: F401

__all__ = [
    "__version__",
    "ALPHAS",
    "BENCH_VEC_PARAMS",
    "SEED",
    "TASKS",
    "TEST_SIZE",
    "THRESHOLD_GRID",
    "VAL_SIZE",
    "VEC_PARAMS",
    "clean_text",
    "evaluate",
    "load_all",
    "load_task",
    "make_pipeline",
    "model_size_kb",
    "pick_threshold",
    "prepare_all",
    "train_all",
    "tune_alpha",
]
