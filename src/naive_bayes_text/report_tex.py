from __future__ import annotations
import json
from datetime import date
from pathlib import Path

import pandas as pd

from .config import (
    ALPHAS,
    DATASET_CITATIONS,
    DOCS_DIR,
    REPORTS_DIR,
    THRESHOLD_GRID,
    VAL_SIZE,
)

TEMPLATE_PATH = DOCS_DIR / "report_template.tex"

TASK_TITLE = {
    "sms": r"SMS Spam Collection",
    "imdb": r"IMDb Sentiment",
}

#: The one number that cannot be re-measured: the F1 we got by choosing the
#: threshold on the test set, which is exactly the mistake the report withdraws.
#: Re-deriving it would mean repeating the leak on purpose, so it stays pinned
#: here with its provenance rather than hidden in prose.
WITHDRAWN_LEAKED_F1 = 0.9441    # threshold selected on the test set

#: The alpha=1 baseline and the latency split are NOT pinned: the pipeline
#: measures both (experiments.run_untuned_baseline / run_latency_split) and
#: reports/metrics.json carries them, so the prose cannot drift.

#: Verdict text per rejected variant. The notebook uses the same strings.
VERDICTS = {
    "no_cap": r"Rejected: 380\,000+ features and no gain. "
              r"The cap is what bounds memory and latency.",
    "min_df_1": r"Rejected: the sign of the effect flips between training "
                r"subsets, which is noise rather than a real effect.",
    "max_features_100k": r"Rejected: indistinguishable from the 50k cap.",
    "shipped": r"\textbf{Shipped.}",
    "ngram_13_sms": r"Rejected: the gain does not survive on the other task.",
    "ngram_13_imdb": r"Rejected: a real <<GAIN>> F1 <<UNIT>> on IMDb, but it "
                     r"costs about the same on SMS. Recorded as a deliberate "
                     r"trade-off, not an oversight.",
}



# LaTeX helpers

def esc(text) -> str:
    """Escape the LaTeX special characters that occur in our strings."""
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def num(value, digits: int = 4, dash: str = "--") -> str:
    """Format a number for a table cell; placeholder if missing."""
    if value is None:
        return dash
    try:
        f = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if f != f:  # NaN
        return dash
    return f"{f:.{digits}f}"


def us(value, digits: int = 1) -> str:
    """Milliseconds, rendered as microseconds when very small."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "--"
    if f != f:
        return "--"
    if f < 0.001:
        return f"{f * 1000:.{digits}f}~\\mu s"
    return f"{f:.3f}~ms"


def table(rows: list[list[str]], header: list[str], widths: str | None = None) -> str:
    """Render a booktabs table, or a longtable when ``widths`` is given."""
    if widths is None:
        spec = "l" + "r" * (len(header) - 1)
        return "\n".join([
            r"\begin{tabular}{" + spec + "}",
            r"\toprule",
            " & ".join(header) + r" \\",
            r"\midrule",
            *[" & ".join(row) + r" \\" for row in rows],
            r"\bottomrule",
            r"\end{tabular}",
        ])
    return "\n".join([
        r"\begin{longtable}{" + widths + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        r"\endhead",
        *[" & ".join(row) + r" \\" for row in rows],
        r"\bottomrule",
        r"\end{longtable}",
    ])


def _label(text: str) -> str:
    """Bold whichever marker a configuration label carries."""
    return esc(text).replace("SHIPPED", r"\textbf{SHIPPED}") \
                     .replace("FINAL", r"\textbf{FINAL}")


# Token computation

def _pick(frame: pd.DataFrame, task: str, prefix: str, column: str) -> float:
    subset = frame[(frame["task"] == task) &
                   (frame["model"].str.startswith(prefix))]
    return float(subset[column].iloc[0]) if len(subset) else float("nan")


def _row_count(row) -> int:
    """Accept a boolean written as a string by ``to_dict('records')``."""
    if isinstance(row, str):
        return 1 if row.strip().lower() in ("true", "1", "yes") else 0
    return 1 if row else 0


def _verdict(variant: str, task: str, gain_points: float) -> str:
    if "(1,3)" in variant:
        if task == "imdb":
            unit = "point" if abs(gain_points) < 2 else "points"
            return (VERDICTS["ngram_13_imdb"]
                    .replace("<<GAIN>>", f"{gain_points:+.1f}")
                    .replace("<<UNIT>>", unit))
        return VERDICTS["ngram_13_sms"]
    if "min_df=1" in variant:
        return VERDICTS["min_df_1"]
    if "no vocabulary cap" in variant:
        return VERDICTS["no_cap"]
    if "100k" in variant:
        return VERDICTS["max_features_100k"]
    return VERDICTS["shipped"]


def build_tokens(payload: dict) -> dict[str, str]:
    """Compute every ``<<TOKEN>>`` value from the saved measurements."""
    res = payload["results"]
    half = payload["results_at_threshold_0.5"]
    meta = payload["meta"]
    cv = payload["cv_macro_f1"]
    alpha = payload["best_alpha"]

    ablation = pd.DataFrame(payload["ablation"])
    bench = pd.DataFrame(payload["benchmark"])
    variants = pd.DataFrame(payload["improvement_variants"])
    calib = pd.DataFrame(payload["calibration"])

    # Measured, not remembered: the alpha=1 baseline and the single-call latency
    # split are both produced by the pipeline and stored in metrics.json.
    untuned = payload["untuned_baseline"]["sms"]
    lat = payload["latency_split"]["sms"]

    # main table
    main_rows = [[
        TASK_TITLE[task], f"{meta[task]['n_train']:,}", f"{meta[task]['n_test']:,}",
        f"{meta[task]['n_features']:,}", f"{meta[task]['best_alpha']:g}",
        f"{meta[task]['threshold']:g}", num(res[task]["accuracy"]),
        num(res[task]["precision"]), num(res[task]["recall"]), num(res[task]["f1"]),
        num(half[task]["f1"]), num(res[task]["f1_macro"]), num(res[task]["roc_auc"]),
    ] for task in res]

    headline_rows = [[
        TASK_TITLE[task], f"{meta[task]['n_train']:,}", f"{meta[task]['n_test']:,}",
        f"{meta[task]['n_features']:,}", f"{meta[task]['best_alpha']:g}",
        f"{meta[task]['threshold']:g}", num(res[task]["accuracy"]),
        num(res[task]["precision"]), num(res[task]["recall"]), num(res[task]["f1"]),
        num(res[task]["roc_auc"]),
    ] for task in res]

    #  ablation table
    ablation_rows = [[
        TASK_TITLE.get(row["task"], row["task"]), _label(row["config"]),
        f"{int(row['vocab']):,}", num(row["accuracy"]), num(row["f1"]),
        num(row["f1_macro"]),
    ] for _, row in ablation.iterrows()]

    #  benchmark tables
    bench_header = ["Model", "Fit (s)", "Tune (s)", r"Infer", "Size (KB)",
                    "Accuracy", "Precision", "Recall", "F1", "ROC--AUC"]

    def bench_table(task: str) -> str:
        rows = []
        for _, row in bench[bench["task"] == task].iterrows():
            if str(row.get("note", "")):
                rows.append([esc(row["model"])] +
                            [r"\emph{" + esc(row["note"]) + "}"] * 9)
                continue
            rows.append([
                esc(row["model"]), num(row["fit_s"], 3), num(row["tune_s"], 2),
                us(row["infer_ms"]), num(row["size_kb"], 1), num(row["accuracy"]),
                num(row["precision"]), num(row["recall"]), num(row["f1"]),
                num(row["roc_auc"]),
            ])
        return table(rows, bench_header)

    # variant table
    variant_rows = []
    for _, row in variants.iterrows():
        task = row["task"]
        shipped = variants[(variants["task"] == task) &
                           (variants["is_shipped"].map(_row_count) == 1)]
        base_f1 = float(shipped["f1_test"].iloc[0]) if len(shipped) else float("nan")
        # F1 is stored as a fraction; every human-facing figure is in points.
        gain_points = (float(row["f1_test"]) - base_f1) * 100
        variant_rows.append([
            TASK_TITLE.get(task, task), _label(row["variant"]),
            num(row["f1_val"]), num(row["f1_test"]),
            _verdict(row["variant"], task, gain_points),
        ])
    variant_table = table(
        variant_rows, ["Variant", "F1 val", "F1 test", "Verdict"],
        widths=r"@{}p{0.22\textwidth}rrp{0.52\textwidth}@{}",
    )

    # scalar figures
    imdb_ship = float(ablation[(ablation["task"] == "imdb") &
                               (ablation["role"] == "shipped")]["f1"].iloc[0])
    imdb_uni = float(ablation[(ablation["task"] == "imdb") &
                              (ablation["config"] == "TF-IDF, unigram")]["f1"].iloc[0])
    graphs = variants[(variants["task"] == "imdb") &
                       (variants["variant"].str.contains(r"\(1,3\)"))]
    graphs_base = variants[(variants["task"] == "imdb") &
                           (variants["is_shipped"].map(_row_count) == 1)]
    graphs_gain = ((float(graphs["f1_test"].iloc[0]) -
                    float(graphs_base["f1_test"].iloc[0])) * 100
                   if len(graphs) and len(graphs_base) else 0.0)
    brier = calib[calib["task"] == "imdb"].iloc[0]

    nb_fit = _pick(bench, "sms", "MultinomialNB", "fit_s")
    nb_tune = _pick(bench, "sms", "MultinomialNB", "tune_s")
    nb_infer = _pick(bench, "sms", "MultinomialNB", "infer_ms")
    nb_size = _pick(bench, "sms", "MultinomialNB", "size_kb")
    lr_fit = _pick(bench, "sms", "Logistic", "fit_s")
    lr_imdb_f1 = _pick(bench, "imdb", "Logistic", "f1")
    knn_infer = _pick(bench, "sms", "KNN", "infer_ms")
    rf_size = _pick(bench, "sms", "RandomForest", "size_kb")

    from .config import DATA_RAW
    sms_raw = pd.read_csv(DATA_RAW / "sms_spam.csv")
    trivial_acc = float((sms_raw["label"] == "ham").mean())

    leftover_unit = "point" if abs(graphs_gain) < 2 else "points"

    return {
        "DATE": date.today().strftime("%d %B %Y"),
        "ABSTRACT_TEXT": _abstract_text(),
        "HEADLINE_TABLE": table(headline_rows, [
            "Dataset", "Train", "Test", "Features", r"$\alpha$", "Cut-off",
            "Accuracy", "Precision", "Recall", "F1", "ROC--AUC"]),
        "MAIN_TABLE": table(main_rows, [
            "Dataset", "Train", "Test", "Features", r"$\alpha$", "Cut-off",
            "Accuracy", "Precision", "Recall", "F1", "F1 @ 0.5", "Macro-F1",
            "ROC--AUC"]),
        "ABLATION_TABLE": table(ablation_rows, [
            "Dataset", "Configuration", "Features", "Accuracy", "F1", "Macro-F1"]),
        "BENCH_SMS": bench_table("sms"),
        "BENCH_IMDB": bench_table("imdb"),
        "VARIANT_TABLE": variant_table,
        "ALPHAS": str(list(ALPHAS)),
        "ALPHA_SMS": f"{alpha['sms']:g}",
        "ALPHA_IMDB": f"{alpha['imdb']:g}",
        "ALPHA_FACTOR": f"{alpha['imdb'] / alpha['sms']:.0f}",
        "VAL_PCT": f"{VAL_SIZE * 100:.0f}",
        "THR_FIRST": f"{THRESHOLD_GRID[0]:g}",
        "THR_LAST": f"{THRESHOLD_GRID[-1]:g}",
        "TRIVIAL_ACC": f"{trivial_acc:.4f}",
        "LEAKED_F1": f"{WITHDRAWN_LEAKED_F1:.4f}",
        "UNTUNED_F1": f"{untuned['f1']:.4f}",
        "UNTUNED_RECALL": f"{untuned['recall']:.2f}",
        "SMS_MISSED": f"{untuned['n_missed']}",
        "SMS_TOTAL": f"{untuned['n_positive']}",
        "SMS_ACC": f"{res['sms']['accuracy']:.4f}",
        "SMS_F1": f"{res['sms']['f1']:.4f}",
        "SMS_RECALL": f"{res['sms']['recall']:.4f}",
        "SMS_AUC": f"{res['sms']['roc_auc']:.4f}",
        "F1_GAIN": f"{(res['sms']['f1'] - untuned['f1']) * 100:.0f}",
        "NB_E2E_MS": f"{lat['MultinomialNB_e2e_ms']:.2f}",
        "LINEAR_E2E_MS": f"{max(lat['LinearSVC_e2e_ms'], lat['LogisticRegression_e2e_ms']):.2f}",
        "VEC_SHARE_MS": f"{lat['vectorise_ms']:.2f}",
        "CV_SMS": f"{sum(cv['sms']) / len(cv['sms']):.4f}",
        "CV_IMDB": f"{sum(cv['imdb']) / len(cv['imdb']):.4f}",
        "ABLATION_DELTA": f"{(imdb_ship - imdb_uni) * 100:.1f} F1 points",
        "NB_SMS_FIT": f"{nb_fit:.3f}",
        "NB_TUNE_SMS": f"{nb_tune:.2f}",
        "NB_INFER": us(nb_infer),
        "KNN_INFER": us(knn_infer, 0),
        "LOGREG_SMS_FIT": f"{lr_fit:.3f}",
        "SPEED_RATIO": f"{lr_fit / nb_fit:.0f}",
        "RF_SIZE_RATIO": f"{rf_size / nb_size:.1f}",
        "LOGREG_IMDB_F1": f"{lr_imdb_f1:.4f}",
        "NB_IMDB_F1": f"{res['imdb']['f1']:.4f}",
        "IMDB_GAP": f"{(lr_imdb_f1 - res['imdb']['f1']) * 100:.1f}",
        "GRAPHS_GAIN": f"{graphs_gain:+.1f}",
        "BRIER_RAW": f"{float(brier['brier_raw']):.4f}",
        "BRIER_CAL": f"{float(brier['brier_cal']):.4f}",
        "CONF_HIGH": f"{untuned['overconfident_share'] * 100:.0f}\\,\\%",
        "LEFTOVER": f"{graphs_gain:.1f} F1 {leftover_unit}",
        "CITE_SMS": esc(DATASET_CITATIONS["sms"][1]),
        "CITE_IMDB": esc(DATASET_CITATIONS["imdb"][1]),
    }


def _abstract_text() -> str:
    return (
        "This report documents the experimental work behind a Naive Bayes text "
        "classifier built for two real applications: SMS spam filtering and "
        "movie-review sentiment analysis. We use \\code{MultinomialNB} from "
        "scikit-learn on top of a TF--IDF representation, train on two standard "
        "English benchmarks, and report accuracy, precision, recall, F1 and "
        "ROC--AUC together with training time, inference latency and model size."
        "\n\nThe two things we would most like a reviewer to take away are "
        "methodological. First, the decision threshold and the smoothing constant "
        "are selected on data disjoint from the test set --- a \\emph{validation "
        "slice carved out of the training set} --- and the test set is read exactly "
        f"once. An earlier revision of this work swept the threshold on the test "
        f"set and reported an F1 of {WITHDRAWN_LEAKED_F1:.4f}; that figure was "
        "optimistic and has been withdrawn. Second, the speed advantage of Naive "
        "Bayes is real but narrower than it first appears: its training step is "
        "several hundred times cheaper than Logistic Regression, yet end-to-end "
        "single-message latency is dominated by the shared TF--IDF step, so Naive "
        "Bayes is not in fact the fastest model at inference."
    )


# Rendering

def render(payload: dict | None = None) -> str:
    """Return the rendered LaTeX, raising if any placeholder is left."""
    if payload is None:
        payload = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    for key, value in build_tokens(payload).items():
        text = text.replace(f"<<{key}>>", str(value))
    leftovers = sorted({text[i:i + 40].split(">>")[0] + ">>"
                        for i in range(len(text)) if text.startswith("<<", i)})
    if leftovers:
        raise RuntimeError(f"unresolved LaTeX placeholders: {leftovers}")
    return text


def write_tex_report(path: Path | None = None, verbose: bool = True) -> str:
    """Render ``docs/report.tex`` from the saved measurements."""
    path = path or (DOCS_DIR / "report.tex")
    text = render()
    path.write_text(text, encoding="utf-8")
    if verbose:
        print(f"wrote {path}  ({len(text.splitlines())} lines, "
              f"{text.count('{')}/{text.count('}')} braces)")
    return str(path)
