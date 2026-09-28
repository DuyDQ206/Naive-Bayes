from __future__ import annotations

import io
import time
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from .config import ALPHAS, TIMING_REPS, VEC_PARAMS


# Construction

def make_pipeline(vec_params: dict | None = None, alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**(VEC_PARAMS if vec_params is None else vec_params))),
        ("nb", MultinomialNB(alpha=alpha)),
    ])


def tune_alpha(X_train, y_train, vec_params: dict | None = None, verbose: bool = True):
    """Pick the smoothing constant by 3-fold CV inside the training set.

    ``n_jobs=1`` on purpose: shipping 25k raw strings to worker processes costs
    more memory than the fit itself, and MultinomialNB trains in seconds anyway.
    """
    grid = GridSearchCV(
        make_pipeline(vec_params), {"nb__alpha": ALPHAS},
        cv=3, scoring="f1_macro", n_jobs=1,
    )
    grid.fit(X_train, y_train)
    best = grid.best_params_["nb__alpha"]
    if verbose:
        scores = {
            f"{a:g}": round(float(s), 4)
            for a, s in zip(grid.cv_results_["param_nb__alpha"],
                            grid.cv_results_["mean_test_score"])
        }
        vocab = len(grid.best_estimator_["tfidf"].vocabulary_)
        print(f"        best alpha = {best}  (macro-F1 {grid.best_score_:.4f})")
        print(f"        macro-F1 by alpha: {scores}")
        print(f"        vocabulary size  : {vocab}")
    return best, grid

# Inference helpers

def model_size_kb(obj) -> float:
    """Serialised size in KB.

    ``joblib.dump`` returns ``None`` when the target is a file object, so the
    stream position is what actually tells us the size.
    """
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return buffer.tell() / 1024


def positive_proba(pipe: Pipeline, X, positive: str) -> np.ndarray:
    """Probability of the positive class for each row of ``X``."""
    classes = list(pipe.classes_)
    if classes.index(positive) != len(classes) - 1:
        raise ValueError("positive class must be the last one (classes are sorted)")
    return pipe.predict_proba(X)[:, len(classes) - 1]


def labels_at_threshold(pipe: Pipeline, X, positive: str, threshold: float) -> np.ndarray:
    """Apply a decision cut-off instead of a plain argmax.

    ``positive_proba`` also validates that the positive class is the last one, so
    it is called exactly once and its result reused -- calling it twice would
    predict twice over the whole evaluation set.
    """
    classes = np.asarray(pipe.classes_)
    proba = positive_proba(pipe, X, positive)
    return classes[(proba >= threshold).astype(int)]


# Evaluation

def evaluate(pipe: Pipeline, X_train, y_train, X_test, y_test, positive: str,
             threshold: float = 0.5, name: str = "MultinomialNB",
             timing_reps: int = TIMING_REPS) -> tuple[dict, np.ndarray]:
    """Fit, time and score one pipeline. Returns ``(metrics, predictions)``."""
    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_s = time.perf_counter() - start

    y_pred = labels_at_threshold(pipe, X_test, positive, threshold)
    proba = pipe.predict_proba(X_test)

    # Single-sample latency: what an interactive UI actually pays.
    single = X_test.iloc[0] if hasattr(X_test, "iloc") else X_test[0]
    start = time.perf_counter()
    for _ in range(timing_reps):
        pipe.predict([single])
    infer_ms = (time.perf_counter() - start) / timing_reps * 1000

    metrics = {
        "model": name,
        "threshold": float(threshold),
        "train_s": train_s,
        "infer_ms": infer_ms,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=positive, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=positive, zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label=positive, zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": (roc_auc_score((y_test == positive).astype(int),
                                  proba[:, list(pipe.classes_).index(positive)])
                    if hasattr(pipe, "predict_proba") else float("nan")),
        "size_kb": model_size_kb(pipe),
    }
    return metrics, y_pred


def pick_threshold(X_train_fit, y_train_fit, X_val, y_val, alpha: float,
                   positive: str, vec_params: dict | None = None) -> tuple[float, list[dict]]:

    from .config import THRESHOLD_GRID

    pipe = make_pipeline(vec_params, alpha=alpha).fit(X_train_fit, y_train_fit)
    proba = positive_proba(pipe, X_val, positive)
    y_bin = (y_val == positive).astype(int).to_numpy()

    rows = []
    for threshold in THRESHOLD_GRID:
        y_hat = (proba >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": precision_score(y_bin, y_hat, zero_division=0),
            "recall": recall_score(y_bin, y_hat, zero_division=0),
            "f1": f1_score(y_bin, y_hat, zero_division=0),
        })
    best = max(rows, key=lambda r: r["f1"])
    return float(best["threshold"]), rows
