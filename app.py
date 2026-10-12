import os
import glob
import time
import joblib
import pandas as pd
import streamlit as st

# ============================================================
# 1. Page configuration
# ============================================================
st.set_page_config(
    page_title="Naive Bayes Application",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# High-contrast theme for presentations
# ============================================================
st.markdown(
    """
    <style>
    /* Main canvas and default text */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF;
        color: #172033;
    }
    .stApp, .stApp p, .stApp li, .stApp label,
    .stApp span, .stApp div, .stApp td, .stApp th {
        color: #172033;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
        font-weight: 700 !important;
    }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #4B5563 !important;
    }
    /* Sidebar: clear separation and readable controls */
    [data-testid="stSidebar"] {
        background-color: #F1F5F9;
        border-right: 1px solid #CBD5E1;
    }
    [data-testid="stSidebar"] * {
        color: #172033 !important;
    }
    /* Inputs */
    input, textarea, [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border-color: #64748B !important;
    }
    input::placeholder, textarea::placeholder {
        color: #64748B !important;
        opacity: 1 !important;
    }
    /* Buttons: dark text on a light surface */
    .stButton button, [data-testid="stDownloadButton"] button {
        background-color: #FFFFFF !important;
        color: #123B6D !important;
        border: 1px solid #345B86 !important;
        font-weight: 600 !important;
    }
    .stButton button:hover, [data-testid="stDownloadButton"] button:hover {
        background-color: #E8F1FB !important;
        border-color: #123B6D !important;
    }
    /* Tabs and tables */
    [data-baseweb="tab"] { color: #334155 !important; }
    [data-baseweb="tab"][aria-selected="true"] {
        color: #123B6D !important;
        font-weight: 700 !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #CBD5E1;
        border-radius: 6px;
    }
    [data-testid="stAlert"] { border: 1px solid #64748B; }
    /* Make secondary/muted text readable on a projector */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    .stMarkdown small, small,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stHelp"],
    [data-testid="stTooltipContent"] {
        color: #27364B !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    .stMarkdown p, .stMarkdown li, .stMarkdown span,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stExpander"] summary,
    [data-testid="stDataFrame"] * {
        color: #172033;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 700 !important;
    }
    [data-testid="stExpander"] summary {
        font-weight: 700 !important;
    }
    [data-testid="stDataFrame"] th {
        color: #111827 !important;
        font-weight: 700 !important;
    }
    [data-testid="stDataFrame"] td {
        color: #172033 !important;
        font-weight: 500 !important;
    }
    /* Stronger visibility for status and helper text */
    [data-testid="stAlert"] p { font-weight: 600 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
BENCHMARK_DIR = os.path.join(MODELS_DIR, "benchmark")

IMDB_FILES = {
    "Multinomial Naive Bayes": "imdb_multinomial_nb.joblib",
    "Bernoulli Naive Bayes": "imdb_bernoulli_nb.joblib",
    "Complement Naive Bayes": "imdb_complement_nb.joblib",
    "Logistic Regression": "imdb_logistic_regression.joblib",
    "Linear SVC": "imdb_linear_svc.joblib",
}

SMS_FILES = {
    "Multinomial Naive Bayes": "sms_multinomial_nb.joblib",
    "Bernoulli Naive Bayes": "sms_bernoulli_nb.joblib",
    "Complement Naive Bayes": "sms_complement_nb.joblib",
    "Logistic Regression": "sms_logistic_regression.joblib",
    "Linear SVC": "sms_linear_svc.joblib",
    "K-Nearest Neighbors": "sms_knn.joblib",
    "Random Forest": "sms_random_forest.joblib",
}

ALL_MODELS_OPTION = "All Models"

TASKS = {
    "IMDb Sentiment Analysis": {
        "folder": "IMDb Sentiment",
        "files": IMDB_FILES,
        "positive_labels": {"pos", "positive", "true"},
        "positive_text": "Positive",
        "negative_text": "Negative",
        "description": "Classifies movie reviews as positive or negative.",
    },
    "SMS Spam Detection": {
        "folder": "SMS Spam",
        "files": SMS_FILES,
        "positive_labels": {"spam", "true"},
        "positive_text": "Spam",
        "negative_text": "Ham (Legitimate Message)",
        "description": "Detects spam, promotional, or potentially fraudulent text messages.",
    },
}

SAMPLES = {
    "IMDb Sentiment Analysis": {
        "Positive Review": (
            "An absolute masterpiece! Stunning visuals, brilliant acting, "
            "and an emotional story."
        ),
        "Negative Review": (
            "Wasted two hours of my life. Boring plot, terrible acting, "
            "and confusing ending."
        ),
    },
    "SMS Spam Detection": {
        "Legitimate Message": (
            "Hey, are we still meeting for lunch today at 12:30?"
        ),
        "Spam Message": (
            "WINNER!! You have won a $1000 gift card! Claim now by clicking "
            "here or call 800-123456"
        ),
    },
}

# ============================================================
# 2. Model loading
# The .joblib files in models/benchmark/ are dictionaries containing
# a Pipeline under the key "model", not bare estimators.
# ============================================================
def find_model_file(filename):
    """Look for a model file in models/benchmark/, then models/, then the project root."""
    for folder in (BENCHMARK_DIR, MODELS_DIR, BASE_DIR):
        candidate = os.path.join(folder, filename)
        if os.path.isfile(candidate):
            return candidate
    
    # Last resort: search subfolders of models/benchmark/ (e.g. benchmark/imdb/...).
    matches = glob.glob(os.path.join(BENCHMARK_DIR, "**", filename), recursive=True)
    return matches[0] if matches else None

@st.cache_resource

def load_all_models():
    loaded = {task: {} for task in TASKS}
    errors = []
    for task_name, config in TASKS.items():
        for display_name, filename in config["files"].items():
            path = find_model_file(filename)
            if path is None:
                continue
            try:
                artifact = joblib.load(path)
                if isinstance(artifact, dict) and "model" in artifact:
                    model = artifact["model"]
                    metrics = artifact.get("metrics", {})
                    label_map = artifact.get("label_map", {})
                    meta = artifact.get("meta", {})
                    classes = artifact.get("classes", [])
                else:
                    # Support a bare estimator saved with joblib.
                    model = artifact
                    metrics = {}
                    label_map = {}
                    meta = {}
                    classes = list(getattr(model, "classes_", []))

                loaded[task_name][display_name] = {
                    "model": model,
                    "metrics": metrics,
                    "label_map": label_map,
                    "meta": meta,
                    "classes": classes,
                    "path": path,
                }
            except Exception as exc:
                errors.append(f"{filename}: {exc}")
    return loaded, errors

all_models, load_errors = load_all_models()

# ============================================================
# 3. Helpers
# ============================================================
def get_label_text(prediction, task_name, label_map):
    """Convert a model label to UI text without assuming numeric label encoding.
    Numeric labels (0/1) are interpreted through the artifact's label_map first.
    If no mapping exists, their meaning is ambiguous, so return an unknown status
    rather than incorrectly treating 1 as positive or 0 as negative.
    """
    task_config = TASKS[task_name]
    raw = str(prediction).strip()
    normalized = raw.lower()
    def classify_label(label):
        value = str(label).strip().lower()
        if task_name == "IMDb Sentiment Analysis":
            if value in {"pos", "positive", "1", "true"} or "positive" in value:
                return task_config["positive_text"], True
            if value in {"neg", "negative", "0", "false"} or "negative" in value:
                return task_config["negative_text"], False
        else:
            if value in {"spam", "1", "true"} or ("spam" in value and "not spam" not in value):
                return task_config["positive_text"], True
            if value in {"ham", "legit", "legitimate", "not spam", "0", "false"}:
                return task_config["negative_text"], False
        return None
    # Explicit text labels are safe to interpret directly.
    direct = classify_label(normalized)
    if direct is not None and not normalized.replace(".", "", 1).isdigit():
        return direct
    # Resolve numeric / encoded labels from label_map before interpreting them.
    mapped = None
    if isinstance(label_map, dict):
        candidates = [raw]
        try:
            numeric = int(float(raw))
            candidates.extend([numeric, str(numeric)])
        except (ValueError, TypeError):
            pass
        for key in candidates:
            if key in label_map:
                mapped = label_map[key]
                break
        # Also support the inverse format, e.g. {"ham": 0, "spam": 1}.
        if mapped is None:
            try:
                numeric_prediction = int(float(raw))
                for label_name, encoded_value in label_map.items():
                    if str(encoded_value).strip() == str(numeric_prediction):
                        mapped = label_name
                        break
            except (ValueError, TypeError):
                pass
    if mapped is not None:
        mapped_result = classify_label(mapped)
        if mapped_result is not None:
            return mapped_result
        # Preserve the mapped label, but do not falsely claim its polarity.
        return str(mapped), None
    # Unknown numeric encodings must not silently become Positive/Negative/Spam/Ham.
    if normalized in {"0", "1", "0.0", "1.0"}:
        return f"Unknown label ({raw}) — check model label_map", None
    return raw, None

def get_confidence(model, text):
    """Return the largest predicted class probability, if available.

    This value is not guaranteed to be calibrated and should not be interpreted
    as the probability that the prediction is correct.
    """
    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            return float(max(probabilities)) * 100
    except Exception:
        pass
    # decision_function is a score, not a calibrated probability.
    return None

def format_metric(value, percentage=True):
    try:
        number = float(value)
        if number != number:  # NaN
            return "N/A"
        return f"{number * 100:.2f}%" if percentage else f"{number:.4f}"
    except (TypeError, ValueError):
        return "N/A"

def show_result_dialog(task_name, model_name, text_input, result, is_positive,
                       confidence, elapsed_ms):
    """Compact modal dialog for a single selected model."""
    @st.dialog("Analysis Result", width="small")
    def _dialog():
        st.caption(task_name)
        # Highlight the selected model.
        st.markdown(f"Model: :blue-background[**{model_name}**]")
        if is_positive:
            st.success(f"Prediction: **{result}**")
        else:
            st.info(f"Prediction: **{result}**")
        col1, col2 = st.columns(2)
        with col1:
            if confidence is not None:
                st.metric("Estimated Confidence", f"{confidence:.2f}%")
                st.caption("Highest class probability returned by the model; not necessarily a calibrated probability or a guarantee of correctness.")
            else:
                st.metric("Estimated Confidence", "N/A")
                st.caption("This model does not provide class probabilities.")
        with col2:
            st.metric("Inference Time", f"{elapsed_ms:.2f} ms")
        if confidence is not None:
            st.progress(min(max(confidence / 100, 0.0), 1.0))
        else:
            st.caption("This model does not provide class probabilities. Confidence is unavailable.")
        with st.expander("Analyzed Text"):
            st.write(text_input)
        if st.button("Close", use_container_width=True):
            st.rerun()
    _dialog()

def show_all_results_dialog(task_name, text_input, rows):
    """Modal dialog listing the predictions of every available model."""
    @st.dialog("All Models Results", width="large")
    def _dialog():
        st.caption(task_name)
        # Use the majority prediction as the final result, rather than always
        # counting only the positive class. Ignore failed/unknown predictions.
        valid = [r for r in rows if r["is_positive"] is not None]
        positive_count = sum(1 for r in valid if r["is_positive"])
        negative_count = len(valid) - positive_count
        positive_text = TASKS[task_name]["positive_text"]
        negative_text = TASKS[task_name]["negative_text"]
        if positive_count > negative_count:
            final_label = positive_text
            final_count = positive_count
        elif negative_count > positive_count:
            final_label = negative_text
            final_count = negative_count
        else:
            final_label = None
            final_count = positive_count
        if not valid:
            st.error("No valid model predictions are available to determine a final result.")
        elif final_label is None:
            st.markdown(
                f'<div style="padding:16px 18px;border:2px solid #B45309;'
                f'border-radius:10px;background:#FFFBEB;color:#78350F;font-size:20px;'
                f'font-weight:800;">FINAL RESULT: TIE</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**{positive_count}/{len(valid)}** models predict **{positive_text}**; "
                f"**{negative_count}/{len(valid)}** predict **{negative_text}**."
            )
        else:
            # HAM and POSITIVE are favorable labels (green); SPAM and NEGATIVE are red.
            if task_name == "SMS Spam Detection":
                final_color = "#15803D" if final_label == negative_text else "#B91C1C"
                final_bg = "#F0FDF4" if final_label == negative_text else "#FEF2F2"
            else:
                final_color = "#15803D" if final_label == positive_text else "#B91C1C"
                final_bg = "#F0FDF4" if final_label == positive_text else "#FEF2F2"
            st.markdown(
                f'<div style="padding:16px 18px;border:2px solid {final_color};'
                f'border-radius:10px;background:{final_bg};color:{final_color};'
                f'font-size:22px;font-weight:800;">FINAL RESULT: {final_label.upper()} '
                f'— {final_count}/{len(valid)} MODELS</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**{final_count}/{len(valid)}** models predict **{final_label}**."
            )

        table = pd.DataFrame([
            {
                "Model": r["model"],
                "Prediction": r["result"],
                "Estimated Confidence": (
                    f"{r['confidence']:.2f}%" if r["confidence"] is not None else "N/A"
                ),
                "Inference Time (ms)": (
                    f"{r['elapsed_ms']:.2f}" if r["elapsed_ms"] is not None else "N/A"
                ),
            }
            for r in rows
        ])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            "Estimated confidence is the highest class probability returned by a model "
            "when available. It may not be calibrated and is not a guarantee that a prediction is correct."
        )
        with st.expander("Analyzed Text"):
            st.write(text_input)
        if st.button("Close", use_container_width=True):
            st.rerun()
    _dialog()

# ============================================================
# 4. Sidebar
# ============================================================
with st.sidebar:
    st.title("System Configuration")
    task_name = st.selectbox(
        "Select a Task",
        list(TASKS.keys()),
    )
    available_models = all_models.get(task_name, {})
    run_all = False
    if available_models:
        model_name = st.selectbox(
            "Select a Model",
            [ALL_MODELS_OPTION] + list(available_models.keys()),
            index=0,
        )
        run_all = model_name == ALL_MODELS_OPTION
        if run_all:
            selected_artifact = None
            current_model = None
        else:
            selected_artifact = available_models[model_name]
            current_model = selected_artifact["model"]
    else:
        model_name = None
        selected_artifact = None
        current_model = None
        st.error("No compatible models found. Please check the models/benchmark/ directory.")
    st.divider()
    st.subheader("How to Use")
    st.markdown(
        """
        1. Select a task above.
        2. Select a model, or keep `All Models` to run every available model.
        3. Enter text or choose a sample in the main area.
        4. Click **Analyze Text**.
        5. Review the results in the dialog that appears.
        """
    )
    st.divider()
    st.caption("Naive Bayes Application")

# ============================================================
# 5. Main interface
# ============================================================
st.title(
    "IMDb Movie Review Sentiment Analyzer"
    if task_name == "IMDb Sentiment Analysis"
    else "SMS Spam Detection System"
)
st.write(TASKS[task_name]["description"])

tab_predict, tab_models, tab_about = st.tabs(
    ["Predict", "Available Models", "System Information"]
)

with tab_predict:
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("Enter Text")
        st.caption("Enter English text or choose a sample.")
        sample_names = list(SAMPLES[task_name].keys())
        sample_choice = st.selectbox(
            "Quick Sample",
            ["No Sample"] + sample_names,
        )

        default_text = (
            SAMPLES[task_name][sample_choice]
            if sample_choice != "No Sample"
            else ""
        )

        # Keep the chosen sample in session state so the text area is populated.
        if "last_sample_choice" not in st.session_state:
            st.session_state.last_sample_choice = sample_choice
        if st.session_state.last_sample_choice != sample_choice:
            st.session_state.last_sample_choice = sample_choice
            st.session_state.input_text = default_text
        if "input_text" not in st.session_state:
            st.session_state.input_text = default_text

        user_text = st.text_area(
            "Text to Analyze",
            key="input_text",
            height=210,
            placeholder="Enter a movie review or text message here...",
        )

        predict_clicked = st.button(
            "Analyze Text",
            type="primary",
            use_container_width=True,
            disabled=not available_models,
        )

        if predict_clicked:
            text = user_text.strip()
            if not text:
                st.warning("Please enter some text before running the analysis.")
            elif run_all:
                rows = []
                for name, artifact in available_models.items():
                    try:
                        start_time = time.perf_counter()
                        prediction = artifact["model"].predict([text])[0]
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        result_text, is_positive = get_label_text(
                            prediction, task_name, artifact["label_map"]
                        )
                        rows.append({
                            "model": name,
                            "result": result_text,
                            "is_positive": is_positive,
                            "confidence": get_confidence(artifact["model"], text),
                            "elapsed_ms": elapsed_ms,
                        })
                    except Exception as exc:
                        rows.append({
                            "model": name,
                            "result": f"Error: {exc}",
                            "is_positive": None,
                            "confidence": None,
                            "elapsed_ms": None,
                        })
                st.session_state.all_results = {
                    "task": task_name,
                    "text": text,
                    "rows": rows,
                }
                show_all_results_dialog(task_name, text, rows)
            elif current_model is None:
                st.error("The selected model is not ready.")
            else:
                try:
                    start_time = time.perf_counter()
                    prediction = current_model.predict([text])[0]
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    result_text, is_positive = get_label_text(
                        prediction,
                        task_name,
                        selected_artifact["label_map"],
                    )
                    confidence = get_confidence(current_model, text)
                    show_result_dialog(
                        task_name,
                        model_name,
                        text,
                        result_text,
                        is_positive,
                        confidence,
                        elapsed_ms,
                    )
                except Exception as exc:
                    st.error(
                        "Unable to analyze the text. "
                        "Please check the scikit-learn version and model file."
                    )
                    st.code(str(exc))

    # Selected Model is displayed at the top of the right column.
    with right:
        st.subheader("Selected Model")
        if run_all:
            st.write(f"**Name:** All Models ({len(available_models)})")
            saved = st.session_state.get("all_results")
            has_results = bool(saved) and saved["task"] == task_name
            results_by_model = (
                {r["model"]: r for r in saved["rows"]} if has_results else {}
            )

            summary_rows = []
            for name, artifact in available_models.items():
                metrics = artifact["metrics"]
                row = {"Model": name}
                if has_results:
                    r = results_by_model.get(name)
                    row["Prediction"] = r["result"] if r else "N/A"
                    row["Estimated Confidence"] = (
                        f"{r['confidence']:.2f}%"
                        if r and r["confidence"] is not None
                        else "N/A"
                    )
                    row["Inference Time (ms)"] = (
                        f"{r['elapsed_ms']:.2f}"
                        if r and r["elapsed_ms"] is not None
                        else "N/A"
                    )
                row["Accuracy"] = format_metric(metrics.get("accuracy"))
                row["F1-score"] = format_metric(metrics.get("f1"))
                summary_rows.append(row)

            if has_results:
                st.caption("Most recent results from all models:")
                with st.expander("Analyzed Text"):
                    st.write(saved["text"])
            else:
                st.caption(
                    "No results yet. Click **Analyze Text** to run all models."
                )

            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                hide_index=True,
            )

        elif selected_artifact:
            metrics = selected_artifact["metrics"]
            st.write(f"**Name:** {model_name}")
            st.write(f"**File:** `{os.path.relpath(selected_artifact['path'], BASE_DIR)}`")
            if metrics:
                st.write(f"**Accuracy:** {format_metric(metrics.get('accuracy'))}")
                st.write(f"**F1-score:** {format_metric(metrics.get('f1'))}")
            else:
                st.caption("No benchmark metrics are included with this file.")

with tab_models:
    st.subheader("Available Models")
    rows = []

    for current_task, config in TASKS.items():
        task_models = all_models.get(current_task, {})
        for display_name, filename in config["files"].items():
            artifact = task_models.get(display_name)
            metrics = artifact["metrics"] if artifact else {}
            rows.append({
                "Task": current_task,
                "Model": display_name,
                "Status": "Available" if artifact else "File Missing",
                "Accuracy": format_metric(metrics.get("accuracy")),
                "Precision": format_metric(metrics.get("precision")),
                "Recall": format_metric(metrics.get("recall")),
                "F1-score": format_metric(metrics.get("f1")),
                "Filename": filename,
            })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Metrics are read from metadata saved with the model file; "
        "these are saved benchmark results, not recalculated when the application runs."
    )

with tab_about:
    st.subheader("System Information")
    st.write("This application supports two text classification tasks: IMDb sentiment analysis and SMS spam detection.")
    st.write("Benchmark files are loaded from `models/benchmark/` first.")
    st.write("The model pipeline applies TF-IDF text vectorization before classification.")
    st.write(f"IMDb models loaded: {len(all_models['IMDb Sentiment Analysis'])}")
    st.write(f"SMS models loaded: {len(all_models['SMS Spam Detection'])}")

    if load_errors:
        st.warning("Some model files could not be loaded.")
        for error in load_errors:
            st.text(error)

    if not all_models["IMDb Sentiment Analysis"] and not all_models["SMS Spam Detection"]:
        st.error(
            "No models were loaded. Please check the .joblib files "
            "in the models/benchmark/ directory."
        )