from __future__ import annotations
import json
import time
import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import (
    DATA_DIR,
    LABEL_MAP,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    VEC_PARAMS,
)
from .data import clean_text

#: Sentences for the live demo, plus the edges that must not crash on stage.
DEMO_SENTENCES = {
    "sms": [
        "Congratulations! You have won a FREE 1000 pound gift card. Text CLAIM to 80086 now!",
        "URGENT! Your account has been suspended. Verify your password at http://secure-bank-verify.com now",
        "Free entry into our weekly competition! Just reply YES to join.",
        "Hey are we still meeting for lunch tomorrow at that new place near the station?",
        "Can you pick up milk and bread on your way home? Thanks!",
        "Docu no. 4 received this morning, please confirm if the figures match our records.",
    ],
    "imdb": [
        "An absolute masterpiece from the director; the acting is flawless and I loved every minute of it.",
        "One of the best films I have seen this year. Brilliant script, stunning visuals, wonderful soundtrack.",
        "I did NOT like this film at all. It was a complete waste of two hours of my life.",
        "A dull, predictable mess. The plot made no sense and the dialogue was painful to sit through.",
        "The acting was fine but the story dragged in the middle and the ending felt rushed.",
        "Honestly one of the most boring films I have ever watched.",
    ],
}

EDGE_CASES = ["", "   ", "!!!???...", "a", "12345", "<br />", "\U0001f642\U0001f642", "X" * 5000]


# Model export

def save_bundle(name: str, pipe, metrics: dict, alpha: float, threshold: float,
                n_train: int, n_test: int, verbose: bool = True):
    """Write ``models/<task>_bundle.joblib`` -- what the UI loads.

    The dict is intentionally flat and documented: the UI needs the pipeline,
    the label mapping, the tuned threshold and the headline metrics, nothing else.
    """
    bundle = {
        "task": name,
        "model": pipe,                                  # Pipeline(TfidfVectorizer, NB)
        "threshold": float(threshold),                  # F1-optimal cut-off
        "classes": [str(c) for c in pipe.classes_],
        "label_map": LABEL_MAP[name],
        "metrics": {k: metrics[k] for k in
                    ("accuracy", "precision", "recall", "f1", "f1_macro",
                     "roc_auc", "train_s", "infer_ms", "size_kb")},
        "meta": {
            "sklearn_version": sklearn.__version__,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_features": int(len(pipe["tfidf"].vocabulary_)),
            "best_alpha": float(alpha),
            "threshold": float(threshold),
            "vectorizer_params": {k: (list(v) if isinstance(v, tuple) else v)
                                  for k, v in VEC_PARAMS.items()},
            "seed": SEED,
        },
    }
    path = MODELS_DIR / f"{name}_bundle.joblib"
    joblib.dump(bundle, path, compress=3)
    if verbose:
        print(f"[{name}] -> {path.name} ({path.stat().st_size / 1024:.1f} KB)  "
              f"classes={bundle['classes']}  vocab={bundle['meta']['n_features']}  "
              f"alpha={alpha}  threshold={threshold}  F1={metrics['f1']:.4f}")
    return bundle


def save_plain_pickles(name: str, pipe, verbose: bool = True) -> dict[str, float]:
    """Write separate ``model.pkl`` and ``vectorizer.pkl`` files."""
    model_path = MODELS_DIR / f"{name}_model.pkl"
    vectorizer_path = MODELS_DIR / f"{name}_vectorizer.pkl"
    joblib.dump(pipe.named_steps["nb"], model_path, compress=3)
    joblib.dump(pipe.named_steps["tfidf"], vectorizer_path, compress=3)
    sizes = {"model_kb": model_path.stat().st_size / 1024,
             "vectorizer_kb": vectorizer_path.stat().st_size / 1024}
    if verbose:
        print(f"[{name}] -> {model_path.name} ({sizes['model_kb']:.1f} KB), "
              f"{vectorizer_path.name} ({sizes['vectorizer_kb']:.1f} KB)")
    return sizes


# Demo material

def write_demo_samples(bundles: dict, path=None, verbose: bool = True) -> str:
    """Write ``data/demo_samples.txt`` with the labels the models really produce."""
    path = path or (DATA_DIR / "demo_samples.txt")
    lines = [
        "# Demo sentences for the live presentation.",
        "# Format: <typed text>  ||  <predicted label> (confidence, latency)",
        "# Predictions below were produced by actually running the shipped models.",
        "",
    ]
    for task, sentences in DEMO_SENTENCES.items():
        bundle = bundles[task]
        lines.append(f"--- {task} ---")
        for sentence in sentences:
            start = time.perf_counter()
            proba = bundle["model"].predict_proba([clean_text(sentence)])[0]
            elapsed = (time.perf_counter() - start) * 1000
            index = int((proba >= bundle["threshold"]).argmax())
            lines.append(f"{sentence}  ||  {bundle['label_map'][index]} "
                         f"({proba[index]:.4f}, {elapsed:.1f} ms)")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    if verbose:
        print(f"wrote {path}")
        print("\n".join(lines))

    print("\n=== edge cases (must not crash on stage) ===")
    for task, bundle in bundles.items():
        for text in EDGE_CASES:
            try:
                label = bundle["model"].predict([clean_text(text)])[0]
                print(f"  [{task}] {text[:18]!r:24s} -> {label}")
            except Exception as exc:
                print(f"  [{task}] {text[:18]!r:24s} -> FAILED "
                      f"{type(exc).__name__}: {exc}")
    return str(path)


# Reports

def markdown_table(frame: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                cells.append("n/a" if np.isnan(value) else float_format.format(float(value)))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def write_metrics_report(main: dict, ablation: pd.DataFrame, benchmark: pd.DataFrame,
                         improvements: dict, bundles: dict, untuned: dict,
                         latency: dict, path=None, verbose: bool = True) -> str:
    """Write ``reports/metrics_table.md`` -- the file TV2 needs."""
    path = path or (REPORTS_DIR / "metrics_table.md")
    results = main["results"]
    cv_scores = main["cv_scores"]

    parts = [
        "# Experimental metrics (Naive Bayes / MultinomialNB)",
        "",
        "Generated by `naive_bayes_text` (see `notebooks/naive_bayes_text.ipynb`). "
        "Hand this file to TV2 for the comparison table.",
        f"Random seed: {SEED}. Positive class: spam (SMS), pos (IMDb).",
        "",
        "## 0. How selection was done (no test-set peeking)",
        "",
        "- `alpha` is chosen by 3-fold cross-validation **inside the training set**.",
        "- The decision threshold is chosen on a stratified **15% validation slice of "
        "train**. Sweeping the threshold and keeping the best score measured on the "
        "*test* set leaks test labels into the model choice.",
        "- The test set is read exactly once, to produce the numbers below.",
        "- Variants in section 4 are ranked on that same validation slice; their test "
        "column is printed for information only.",
        "",
        "## 1. Main results (test set, at the validation-chosen threshold)",
        "",
        "| Task | n_train | n_test | vocab | best alpha | threshold | Accuracy | "
        "Precision | Recall | F1 | F1 @ 0.5 | macro-F1 | 5-fold macro-F1 | ROC-AUC | "
        "Train (s) | Infer (ms) | Model (KB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, row in results.items():
        meta = bundles[name]["meta"]
        parts.append(
            "| {} | {} | {} | {} | {:g} | {:g} | {} | {} | {} | {} | {} | {} | "
            "{:.4f} +/- {:.4f} | {} | {:.3f} | {:.3f} | {:.1f} |".format(
                name, meta["n_train"], meta["n_test"], meta["n_features"],
                meta["best_alpha"], meta["threshold"],
                fmt(row["accuracy"]), fmt(row["precision"]), fmt(row["recall"]),
                fmt(row["f1"]), fmt(main["results_at_threshold_0.5"][name]["f1"]),
                fmt(row["f1_macro"]), cv_scores[name].mean(), cv_scores[name].std(),
                fmt(row["roc_auc"]), row["train_s"], row["infer_ms"], row["size_kb"],
            )
        )

    parts += [
        "", "## 2. Ablation (effect of preprocessing choices)", "",
        markdown_table(ablation),
        "", "## 3. Benchmark vs other classifiers", "",
        "`fit_s` is one fit of the final hyper-parameters, so that column compares like "
        "with like. `tune_s` is the extra cost of the hyper-parameter search (only the "
        "Naive Bayes variants were tuned).", "",
        markdown_table(benchmark),
        "", "## 4. Improvement search (measured, then rejected)", "",
        "### 4a. Vocabulary and n-gram variants", "",
        "`f1_val` is what the choice was based on. `f1_test` is reported for information "
        "only.", "",
        markdown_table(improvements["variants"]),
        "", "### 4b. Probability calibration", "",
        markdown_table(improvements["calibration"]),
        "", "### 4c. Decision threshold sweep, measured on validation", "",
        markdown_table(improvements["thresholds"]),
        "", "### Why the shipped configuration is what it is", "",
        "- `(1,3)` n-grams help IMDb but cost about the same amount on SMS, and the two "
        "tasks share one vectoriser in one notebook. Rejected in favour of a single "
        "configuration.",
        "- Lifting `max_features` from 50k to 100k changes almost nothing, and removing "
        "the cap entirely (380k+ features) is no better. The 50k cap bounds memory and "
        "latency.",
        "- `min_df=1` is unstable: the sign of its effect flips between training subsets, "
        "which is noise rather than a real effect. `min_df=2` is kept.",
        "- `CalibratedClassifierCV` improves IMDb probabilities but changes nothing on SMS "
        "and multiplies training time by 5. Documented as a known limitation rather than "
        "shipped, to keep the model a plain MultinomialNB.",
        "- The decision threshold is the one thing that was tuned and shipped: see "
        "`threshold` in the bundle. It is only possible because Naive Bayes returns "
        "probabilities.",
        "", "### Notes for TV2", "",
        "- Every model uses the same train/test split and the same TF-IDF vectoriser "
        "(capped at 20,000 features in this table).",
        "- All three Naive Bayes variants had `alpha` tuned by the same 3-fold grid "
        "search, so the comparison between them is fair.",
        "- KNN and RandomForest were skipped on IMDb: densifying a 25,000 x 20,000 matrix "
        "is too slow and too large for a class demo. Run them on SMS to make the "
        "speed/size point.",
        "- `LinearSVC` has no `predict_proba`, so ROC-AUC is n/a and it cannot be "
        "threshold-tuned.",
        "- SMS is ~87% ham, so per-class F1 for spam matters more than overall accuracy.",
        "",
    ]

    path.write_text("\n".join(parts), encoding="utf-8")
    if verbose:
        print(f"wrote {path}")
    return str(path)


def write_metrics_json(main: dict, ablation: pd.DataFrame, benchmark: pd.DataFrame,
                       improvements: dict, bundles: dict, untuned: dict,
                       latency: dict, path=None, verbose: bool = True) -> str:
    """Write ``reports/metrics.json`` -- the machine-readable source of truth."""
    path = path or (REPORTS_DIR / "metrics.json")
    payload = {
        "seed": SEED,
        "results": main["results"],
        "results_at_threshold_0.5": main["results_at_threshold_0.5"],
        "best_alpha": main["best_alpha"],
        "best_threshold": main["best_threshold"],
        "cv_macro_f1": {k: v.tolist() for k, v in main["cv_scores"].items()},
        "ablation": ablation.to_dict("records"),
        "benchmark": benchmark.astype(str).to_dict("records"),
        "improvement_variants": improvements["variants"].to_dict("records"),
        "calibration": improvements["calibration"].to_dict("records"),
        "threshold_sweep_validation": improvements["thresholds"].to_dict("records"),
        "untuned_baseline": untuned,
        "latency_split": latency,
        "meta": {k: bundles[k]["meta"] for k in bundles},
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if verbose:
        print(f"wrote {path}")
    return str(path)
