"""The experiments behind the report.

Four stages, in the order in which they must run:

1. ``run_main``          -- alpha, threshold, then one pass over the test set
2. ``run_ablation``      -- what each preprocessing choice is worth
3. ``run_benchmark``     -- Naive Bayes against six other classifiers
4. ``run_improvements``  -- what else was tried, and why it was rejected

Stage 1 is the only place the test set is read.
"""
from __future__ import annotations
import gc
import time
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from .config import (
    ALPHAS,
    BENCH_VEC_PARAMS,
    CV_FOLDS,
    HEAVY_ON_IMDB,
    KNN_MAX_TEST,
    TASKS,
    THRESHOLD_GRID,
    VARIANTS,
    VEC_PARAMS,
)
from .data import make_validation_split
from .modeling import (
    evaluate,
    make_pipeline,
    model_size_kb,
    pick_threshold,
    positive_proba,
    tune_alpha,
)

#: Preprocessing ladder for the ablation study. ``role`` is a machine-readable
#: flag so that reports never have to parse the human-readable label.
ABLATION_CONFIGS = [
    ("CountVectorizer (raw counts), unigram", "count", dict(ngram_range=(1, 1)), "variant"),
    ("TF-IDF, unigram", "tfidf", dict(ngram_range=(1, 1)), "variant"),
    ("TF-IDF, unigram + english stopwords", "tfidf",
     dict(ngram_range=(1, 1), stop_words="english"), "variant"),
    ("TF-IDF, (1,2), keep negation  [FINAL]", "tfidf", {}, "shipped"),
]

#: Benchmark candidates. The three Naive Bayes variants get alpha tuned, because
#: comparing a tuned model against an untuned one would be meaningless.
BENCHMARK_CANDIDATES = {
    "MultinomialNB": lambda: MultinomialNB(alpha=1.0),
    "BernoulliNB": lambda: BernoulliNB(alpha=1.0),
    "ComplementNB": lambda: ComplementNB(alpha=1.0),
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000, n_jobs=-1),
    "LinearSVC": lambda: LinearSVC(dual="auto", max_iter=5000),
    "KNN (k=5)": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "RandomForest(50)": lambda: RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                                        random_state=0),
}
NB_VARIANTS = ("MultinomialNB", "BernoulliNB", "ComplementNB")



# Stage 1 -- main result

def run_main(splits: dict, verbose: bool = True) -> dict:
    """Tune alpha (CV on train), pick the threshold (validation), score test once."""
    best_alpha, best_threshold, val_splits = {}, {}, {}
    results, results_at_half, y_pred, cv_scores = {}, {}, {}, {}

    for name, (X_train, X_test, y_train, y_test) in splits.items():
        if verbose:
            print(f"\n########## {name.upper()} ##########")
        positive = TASKS[name]["positive"]

        # --- hyper-parameter, chosen inside train -------------------------- #
        alpha, _ = tune_alpha(X_train, y_train, verbose=verbose)
        best_alpha[name] = alpha

        # --- decision threshold, chosen on a validation slice of train ------ #
        X_fit, X_val, y_fit, y_val = make_validation_split(X_train, y_train)
        val_splits[name] = (X_fit, X_val, y_fit, y_val)
        threshold, _ = pick_threshold(X_fit, y_fit, X_val, y_val, alpha, positive)
        best_threshold[name] = threshold

        # --- the one and only read of the test set -------------------------- #
        pipe = make_pipeline(alpha=alpha)
        metrics, preds = evaluate(pipe, X_train, y_train, X_test, y_test,
                                  positive, threshold=threshold)
        results[name], y_pred[name] = metrics, preds
        results_at_half[name], _ = evaluate(make_pipeline(alpha=alpha), X_train, y_train,
                                            X_test, y_test, positive, threshold=0.5)

        cv_scores[name] = cross_val_score(make_pipeline(), X_train, y_train,
                                          cv=CV_FOLDS, scoring="f1_macro", n_jobs=1)

        if verbose:
            print(f"train/test      : {len(X_train)} / {len(X_test)}")
            print(f"vocabulary size : {len(pipe['tfidf'].vocabulary_)}")
            print(f"best alpha      : {alpha}   (3-fold CV inside train)")
            print(f"threshold       : {threshold}   (validation slice, NOT test)")
            print(f"train time      : {metrics['train_s']:.3f} s")
            print(f"inference       : {metrics['infer_ms']:.3f} ms / sample")
            print(f"model size      : {metrics['size_kb']:.1f} KB")
            print(f"5-fold macro-F1 : {cv_scores[name].mean():.4f} "
                  f"+/- {cv_scores[name].std():.4f}")
            print(f"\nclassification report (positive = {positive!r}):")
            print(classification_report(y_test, preds, digits=4, zero_division=0))
            print("confusion matrix (rows=true, cols=pred), order "
                  f"{sorted(set(y_test))}:\n"
                  f"{confusion_matrix(y_test, preds, labels=sorted(set(y_test)))}")
            print(f"F1 @ {threshold} : {metrics['f1']:.4f}")
            print(f"F1 @ 0.5     : {results_at_half[name]['f1']:.4f}")
            print(f"ROC-AUC       : {metrics['roc_auc']:.4f}")

    return {
        "results": results,
        "results_at_threshold_0.5": results_at_half,
        "y_pred": y_pred,
        "cv_scores": cv_scores,
        "best_alpha": best_alpha,
        "best_threshold": best_threshold,
        "val_splits": val_splits,
    }


def plot_confusion_matrices(splits: dict, y_pred: dict, path=None):
    """Save a two-panel confusion-matrix figure for the report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(splits), figsize=(5.5 * len(splits), 4.2),
                             squeeze=False)
    for ax, name in zip(axes[0], splits):
        ConfusionMatrixDisplay.from_predictions(
            splits[name][3], y_pred[name],
            labels=sorted(set(splits[name][3])), ax=ax, colorbar=False,
        )
        ax.set_title(f"{name.upper()} - MultinomialNB")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        print(f"wrote {path}")
    return fig



# Stage 2 -- ablation

def run_ablation(splits: dict, best_alpha: dict, verbose: bool = True) -> pd.DataFrame:
    """Measure what each preprocessing decision is actually worth."""
    rows = []
    for name, (X_train, X_test, y_train, y_test) in splits.items():
        positive = TASKS[name]["positive"]
        for label, kind, override, role in ABLATION_CONFIGS:
            params = dict(VEC_PARAMS)
            params.update(override)
            if kind == "count":
                # CountVectorizer has no TF-IDF weighting options.
                params.pop("sublinear_tf", None)
                vectorizer = CountVectorizer(**params)
            else:
                vectorizer = TfidfVectorizer(**params)
            A, B = vectorizer.fit_transform(X_train), vectorizer.transform(X_test)
            model = MultinomialNB(alpha=best_alpha[name]).fit(A, y_train)
            pred = model.predict(B)
            rows.append(dict(
                task=name, config=label, role=role,
                vocab=len(vectorizer.vocabulary_),
                accuracy=accuracy_score(y_test, pred),
                f1=f1_score(y_test, pred, pos_label=positive, zero_division=0),
                f1_macro=f1_score(y_test, pred, average="macro", zero_division=0),
            ))
            # Each config holds a full sparse matrix; free it before the next one so
            # the cell fits in a few GB of RAM.
            del A, B, vectorizer, model, pred
            gc.collect()

        # The minority class: SMS is ~87% ham, so does re-weighting rescue spam?
        vectorizer = TfidfVectorizer(**VEC_PARAMS)
        A, B = vectorizer.fit_transform(X_train), vectorizer.transform(X_test)
        weights = compute_sample_weight("balanced", y_train)
        model = MultinomialNB(alpha=best_alpha[name]).fit(A, y_train, sample_weight=weights)
        pred = model.predict(B)
        rows.append(dict(
            task=name, config="TF-IDF (1,2) + balanced class weights", role="variant",
            vocab=len(vectorizer.vocabulary_),
            accuracy=accuracy_score(y_test, pred),
            f1=f1_score(y_test, pred, pos_label=positive, zero_division=0),
            f1_macro=f1_score(y_test, pred, average="macro", zero_division=0),
        ))
        del A, B, vectorizer, model, pred
        gc.collect()

    frame = pd.DataFrame(rows)
    if verbose:
        print("\n=== ablation ===")
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return frame


def run_untuned_baseline(splits: dict, verbose: bool = True) -> dict:
    """Measure the shipped configuration with the library's default ``alpha=1``.

    The report quotes this number to show what tuning bought, so it has to be
    measured by the pipeline rather than remembered. It also carries the two
    descriptive statistics the report quotes in prose: how many positive
    messages the untuned model misses, and how often the model is overconfident.
    """
    out: dict[str, dict] = {}
    for name, (X_train, X_test, y_train, y_test) in splits.items():
        positive = TASKS[name]["positive"]
        vectorizer = TfidfVectorizer(**VEC_PARAMS)
        A, B = vectorizer.fit_transform(X_train), vectorizer.transform(X_test)
        # MultinomialNB() with no argument is exactly alpha=1.0, the default.
        model = MultinomialNB().fit(A, y_train)
        pred = model.predict(B)
        proba = model.predict_proba(B).max(axis=1)
        missed = (y_test == positive) & (pred != positive)
        out[name] = dict(
            alpha=float(model.alpha),
            vocab=len(vectorizer.vocabulary_),
            accuracy=accuracy_score(y_test, pred),
            precision=precision_score(y_test, pred, pos_label=positive, zero_division=0),
            recall=recall_score(y_test, pred, pos_label=positive, zero_division=0),
            f1=f1_score(y_test, pred, pos_label=positive, zero_division=0),
            f1_macro=f1_score(y_test, pred, average="macro", zero_division=0),
            n_test=int(len(y_test)),
            n_positive=int((y_test == positive).sum()),
            n_missed=int(missed.sum()),
            overconfident_share=float((proba >= 0.99).mean()),
        )
        del A, B, vectorizer, model, pred, proba
        gc.collect()
    if verbose:
        print("\n=== untuned baseline (library default alpha=1) ===")
        for name, r in out.items():
            print(f"  {name:5s} f1={r['f1']:.4f} recall={r['recall']:.4f} "
                  f"missed={r['n_missed']}/{r['n_positive']} "
                  f"overconfident={r['overconfident_share']:.0%}")
    return out


def run_latency_split(splits: dict, best_alpha: dict, verbose: bool = True) -> dict:
    """Single-call end-to-end latency, which is what a Streamlit demo pays.

    A batch measurement that transforms once and then predicts many times hides
    the vectorisation cost, which is shared by every model on the same
    vocabulary. These figures used to be typed into the report by hand; measuring
    them keeps the prose honest on whatever machine actually runs the pipeline.
    """
    out: dict[str, dict] = {}
    for name, (X_train, X_test, y_train, _y_test) in splits.items():
        vectorizer = TfidfVectorizer(**VEC_PARAMS)
        A = vectorizer.fit_transform(X_train)
        # one real message, so the measurement is not a best case on empty input
        message = str(np.asarray(X_test).ravel()[0])

        def timed(fn, reps: int) -> float:
            fn()  # warm the code paths before the clock starts
            start = time.perf_counter()
            for _ in range(reps):
                fn()
            return (time.perf_counter() - start) / reps * 1000

        candidates = {
            "MultinomialNB": MultinomialNB(alpha=best_alpha[name]),
            "LinearSVC": LinearSVC(),
            "LogisticRegression": LogisticRegression(max_iter=1000),
        }
        row: dict[str, float] = {}
        for model_name, model in candidates.items():
            model.fit(A, y_train)
            # end-to-end: raw text in, label out, vectorisation included
            row[f"{model_name}_e2e_ms"] = timed(
                lambda m=model, v=vectorizer, t=message: m.predict(v.transform([t])), 200)
        row["vectorise_ms"] = timed(lambda v=vectorizer, t=message: v.transform([t]), 200)
        row["n_messages"] = 1
        out[name] = row
        del A, vectorizer, candidates
        gc.collect()
    if verbose:
        print("\n=== single-call end-to-end latency (vectorisation included) ===")
        for name, r in out.items():
            print(f"  {name:5s} NB={r['MultinomialNB_e2e_ms']:.3f}ms "
                  f"SVC={r['LinearSVC_e2e_ms']:.3f}ms "
                  f"LogReg={r['LogisticRegression_e2e_ms']:.3f}ms "
                  f"vectorise={r['vectorise_ms']:.3f}ms")
    return out



# Stage 3 -- benchmark

def run_benchmark(splits: dict, verbose: bool = True) -> pd.DataFrame:
    """Compare against six other classifiers on an identical split and vectoriser.

    ``fit_s`` is one fit of the final hyper-parameters so the column compares
    like with like; ``tune_s`` reports the extra cost of the search separately.
    """
    rows = []
    for name, (X_train, X_test, y_train, y_test) in splits.items():
        positive = TASKS[name]["positive"]
        vectorizer = TfidfVectorizer(**BENCH_VEC_PARAMS)
        A, B = vectorizer.fit_transform(X_train), vectorizer.transform(X_test)
        if verbose:
            print(f"\n[{name}] features={A.shape[1]}  "
                  f"density={A.nnz / (A.shape[0] * A.shape[1]):.4f}  "
                  f"train={A.shape}  test={B.shape}")

        for model_name, factory in BENCHMARK_CANDIDATES.items():
            if name == "imdb" and model_name in HEAVY_ON_IMDB:
                rows.append(dict(task=name, model=model_name,
                                 note="skipped on IMDb (memory/runtime)"))
                if verbose:
                    print(f"  {model_name:20s} SKIPPED on imdb")
                continue

            B_used, y_used = B, y_test
            if model_name.startswith("KNN") and B.shape[0] > KNN_MAX_TEST:
                sel = np.random.RandomState(0).choice(B.shape[0], KNN_MAX_TEST, replace=False)
                B_used, y_used = B[sel], y_test.iloc[sel].to_numpy()

            try:
                best_params, used, tune_s = {}, "", 0.0
                if model_name in NB_VARIANTS:
                    start = time.perf_counter()
                    grid = GridSearchCV(factory(), {"alpha": ALPHAS}, cv=3,
                                        scoring="f1_macro", n_jobs=1).fit(A, y_train)
                    tune_s = time.perf_counter() - start
                    best_params = grid.best_params_
                    used = f"a={best_params['alpha']:g}"

                start = time.perf_counter()
                model = factory().set_params(**best_params).fit(A, y_train)
                fit_s = time.perf_counter() - start

                start = time.perf_counter()
                pred = model.predict(B_used)
                pred_s = time.perf_counter() - start

                if hasattr(model, "predict_proba"):
                    idx = list(model.classes_).index(positive)
                    auc = roc_auc_score((y_used == positive).astype(int),
                                        model.predict_proba(B_used)[:, idx])
                else:
                    auc = float("nan")

                row = dict(
                    task=name, model=f"{model_name} {used}".strip(),
                    fit_s=fit_s, tune_s=tune_s,
                    infer_ms=pred_s / len(y_used) * 1000,
                    size_kb=model_size_kb(model),
                    accuracy=accuracy_score(y_used, pred),
                    precision=precision_score(y_used, pred, pos_label=positive, zero_division=0),
                    recall=recall_score(y_used, pred, pos_label=positive, zero_division=0),
                    f1=f1_score(y_used, pred, pos_label=positive, zero_division=0),
                    f1_macro=f1_score(y_used, pred, average="macro", zero_division=0),
                    roc_auc=auc, note="",
                )
                rows.append(row)
                if verbose:
                    print(f"  {row['model']:28s} acc={row['accuracy']:.4f} "
                          f"f1={row['f1']:.4f} fit={fit_s:7.3f}s "
                          f"tune={tune_s:6.2f}s {row['size_kb']:8.1f}KB")
                del B_used, model, pred
                gc.collect()
            except Exception as exc:  # keep the table complete rather than aborting
                rows.append(dict(task=name, model=model_name,
                                 note=f"failed: {type(exc).__name__}: {exc}"))
                if verbose:
                    print(f"  {model_name:20s} FAILED {type(exc).__name__}: {exc}")
        del A, B, vectorizer
        gc.collect()

    frame = pd.DataFrame(rows)
    if verbose:
        print()
        print(frame.to_string(index=False))
    return frame



# Stage 4 -- improvement search (ranked on validation, reported on test)

def run_improvements(splits: dict, main: dict, verbose: bool = True) -> dict[str, pd.DataFrame]:
    best_alpha = main["best_alpha"]
    best_threshold = main["best_threshold"]

    # The validation slice is re-derived here rather than carried over from
    # run_main: it is deterministic (same seed), and not holding a second copy of
    # the training text keeps the peak memory down on a small machine.
    val_splits = {name: make_validation_split(splits[name][0], splits[name][2])
                  for name in splits}

    # A. vocabulary and n-gram variants 
    variant_rows = []
    for name, (X_train, X_test, y_train, y_test) in splits.items():
        positive = TASKS[name]["positive"]
        X_fit, X_val, y_fit, y_val = val_splits[name]
        for label, override, is_shipped in VARIANTS:
            params = dict(VEC_PARAMS)
            params.update(override)
            vectorizer = TfidfVectorizer(**params)
            # Fit the vectoriser on the training half only, then score the
            # validation half. Fitting on the validation half would leak it.
            A = vectorizer.fit_transform(X_fit)
            grid = GridSearchCV(MultinomialNB(), {"alpha": ALPHAS}, cv=3,
                                scoring="f1_macro", n_jobs=1).fit(A, y_fit)
            model = grid.best_estimator_
            B_val, B_test = vectorizer.transform(X_val), vectorizer.transform(X_test)
            variant_rows.append(dict(
                task=name, variant=label, is_shipped=is_shipped,
                vocab=len(vectorizer.vocabulary_),
                alpha=grid.best_params_["alpha"],
                f1_val=f1_score(y_val, model.predict(B_val),
                                pos_label=positive, zero_division=0),
                f1_test=f1_score(y_test, model.predict(B_test),
                                 pos_label=positive, zero_division=0),
            ))
            del A, B_val, B_test, vectorizer, model
            gc.collect()
        del X_fit, y_fit
        gc.collect()
    variants = pd.DataFrame(variant_rows)
    if verbose:
        print("\n=== A. vocabulary / n-gram variants ===")
        print(variants.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        # Iterate the tasks actually present, not the global TASKS dict: this
        # function is sometimes called on a subset of tasks.
        for name in variants["task"].unique():
            sub = variants[variants["task"] == name]
            print(f"[{name}] best on validation : "
                  f"{sub.loc[sub['f1_val'].idxmax(), 'variant']}")
            print(f"[{name}] best on test      : "
                  f"{sub.loc[sub['f1_test'].idxmax(), 'variant']}   "
                  f"<- coincidence check, not a choice")

    # B. probability calibration 
    calibration_rows = []
    for name, (X_train, X_test, y_train, y_test) in splits.items():
        positive = TASKS[name]["positive"]
        vectorizer = TfidfVectorizer(**VEC_PARAMS)
        A, B = vectorizer.fit_transform(X_train), vectorizer.transform(X_test)
        raw = MultinomialNB(alpha=best_alpha[name]).fit(A, y_train)
        calibrated = CalibratedClassifierCV(
            MultinomialNB(alpha=best_alpha[name]), cv=5, method="sigmoid").fit(A, y_train)
        y_bin = (y_test == positive).astype(int).to_numpy()
        calibration_rows.append(dict(
            task=name,
            brier_raw=brier_score_loss(y_bin, positive_proba(raw, B, positive)),
            brier_cal=brier_score_loss(y_bin, positive_proba(calibrated, B, positive)),
            f1_raw=f1_score(y_test, raw.predict(B), pos_label=positive, zero_division=0),
            f1_cal=f1_score(y_test, calibrated.predict(B), pos_label=positive, zero_division=0),
        ))
        del A, B, vectorizer, raw, calibrated
        gc.collect()
    calibration = pd.DataFrame(calibration_rows)
    if verbose:
        print("\n=== B. probability calibration ===")
        print(calibration.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # C. the threshold sweep, on validation
    threshold_rows = []
    for name, (X_fit, X_val, y_fit, y_val) in val_splits.items():
        positive = TASKS[name]["positive"]
        pipe = make_pipeline(alpha=best_alpha[name]).fit(X_fit, y_fit)
        proba = positive_proba(pipe, X_val, positive)
        y_bin = (y_val == positive).astype(int).to_numpy()
        for threshold in THRESHOLD_GRID:
            y_hat = (proba >= threshold).astype(int)
            threshold_rows.append(dict(
                task=name, threshold=float(threshold),
                precision=precision_score(y_bin, y_hat, zero_division=0),
                recall=recall_score(y_bin, y_hat, zero_division=0),
                f1=f1_score(y_bin, y_hat, zero_division=0),
            ))
    thresholds = pd.DataFrame(threshold_rows)
    if verbose:
        print("\n=== C. threshold sweep (validation) ===")
        print(thresholds.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"chosen: {best_threshold}")

    del val_splits
    gc.collect()
    return {"variants": variants, "calibration": calibration, "thresholds": thresholds}