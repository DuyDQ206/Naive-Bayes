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

ALL_MODELS_OPTION = "Tất cả"

TASKS = {
    "IMDb Sentiment Analysis": {
        "folder": "IMDb Sentiment",
        "files": IMDB_FILES,
        "positive_labels": {"pos", "positive", "1", "true"},
        "positive_text": "Positive",
        "negative_text": "Negative",
        "description": "Phân loại cảm xúc tích cực hoặc tiêu cực trong bài đánh giá phim.",
    },
    "SMS Spam Detection": {
        "folder": "SMS Spam",
        "files": SMS_FILES,
        "positive_labels": {"spam", "1", "true"},
        "positive_text": "Spam",
        "negative_text": "Ham (tin nhắn bình thường)",
        "description": "Nhận diện tin nhắn rác, quảng cáo hoặc có dấu hiệu lừa đảo.",
    },
}

SAMPLES = {
    "IMDb Sentiment Analysis": {
        "Review tích cực": (
            "An absolute masterpiece! Stunning visuals, brilliant acting, "
            "and an emotional story."
        ),
        "Review tiêu cực": (
            "Wasted two hours of my life. Boring plot, terrible acting, "
            "and confusing ending."
        ),
    },
    "SMS Spam Detection": {
        "Tin nhắn bình thường": (
            "Hey, are we still meeting for lunch today at 12:30?"
        ),
        "Tin nhắn spam": (
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
    """Map model output to a readable label without assuming 0/1 encoding."""
    task_config = TASKS[task_name]
    raw = str(prediction).strip()
    normalized = raw.lower()

    # Most benchmark models predict strings: pos/neg or spam/ham.
    if normalized in task_config["positive_labels"]:
        return task_config["positive_text"], True

    if task_name == "IMDb Sentiment Analysis":
        if normalized in {"neg", "negative", "0", "false"}:
            return task_config["negative_text"], False
    else:
        if normalized in {"ham", "legit", "legitimate", "not spam", "0", "false"}:
            return task_config["negative_text"], False

    # Use label_map when a model uses numeric labels.
    try:
        mapped = label_map.get(int(float(raw)))
        if mapped:
            mapped_lower = str(mapped).lower()
            is_positive = mapped_lower in task_config["positive_labels"]
            if task_name == "IMDb Sentiment Analysis":
                is_positive = "positive" in mapped_lower or mapped_lower == "pos"
            else:
                is_positive = "spam" in mapped_lower and "not spam" not in mapped_lower
            return str(mapped), is_positive
    except (ValueError, TypeError):
        pass

    # Safe fallback: show the actual label returned by the model.
    return raw, False


def get_confidence(model, text):
    """Return confidence only when the estimator exposes a meaningful API."""
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

    @st.dialog("Kết quả phân tích", width="small")
    def _dialog():
        st.caption(task_name)
        # Highlight the selected model.
        st.markdown(f"Mô hình: :blue-background[**{model_name}**]")

        if is_positive:
            st.success(f"Kết quả: **{result}**")
        else:
            st.info(f"Kết quả: **{result}**")

        col1, col2 = st.columns(2)
        with col1:
            if confidence is not None:
                st.metric("Độ tin cậy", f"{confidence:.2f}%")
            else:
                st.metric("Độ tin cậy", "N/A")
        with col2:
            st.metric("Thời gian xử lý", f"{elapsed_ms:.2f} ms")

        if confidence is not None:
            st.progress(min(max(confidence / 100, 0.0), 1.0))
        else:
            st.caption("Mô hình này không cung cấp xác suất đã hiệu chỉnh.")

        with st.expander("Văn bản đã phân tích"):
            st.write(text_input)

        if st.button("Đóng", use_container_width=True):
            st.rerun()

    _dialog()


def show_all_results_dialog(task_name, text_input, rows):
    """Modal dialog listing the predictions of every available model."""

    @st.dialog("Kết quả của tất cả mô hình", width="large")
    def _dialog():
        st.caption(task_name)

        valid = [r for r in rows if r["is_positive"] is not None]
        positive_count = sum(1 for r in valid if r["is_positive"])
        positive_text = TASKS[task_name]["positive_text"]
        st.write(
            f"**{positive_count}/{len(valid)}** mô hình dự đoán là "
            f"**{positive_text}**."
        )

        table = pd.DataFrame([
            {
                "Mô hình": r["model"],
                "Kết quả": r["result"],
                "Độ tin cậy": (
                    f"{r['confidence']:.2f}%" if r["confidence"] is not None else "N/A"
                ),
                "Thời gian (ms)": (
                    f"{r['elapsed_ms']:.2f}" if r["elapsed_ms"] is not None else "N/A"
                ),
            }
            for r in rows
        ])
        st.dataframe(table, use_container_width=True, hide_index=True)

        with st.expander("Văn bản đã phân tích"):
            st.write(text_input)

        if st.button("Đóng", use_container_width=True):
            st.rerun()

    _dialog()


# ============================================================
# 4. Sidebar
# ============================================================
with st.sidebar:
    st.title("Cấu hình hệ thống")
    task_name = st.selectbox(
        "Chọn bài toán",
        list(TASKS.keys()),
    )

    available_models = all_models.get(task_name, {})
    run_all = False
    if available_models:
        model_name = st.selectbox(
            "Chọn mô hình",
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
        st.error("Không tìm thấy mô hình phù hợp. Hãy kiểm tra thư mục models/benchmark/.")

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
    ["Dự đoán", "Danh sách mô hình", "Thông tin hệ thống"]
)

with tab_predict:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Nhập văn bản")
        st.caption("Bạn có thể nhập văn bản tiếng Anh hoặc chọn một mẫu thử.")

        sample_names = list(SAMPLES[task_name].keys())
        sample_choice = st.selectbox(
            "Mẫu thử nhanh",
            ["Không sử dụng mẫu"] + sample_names,
        )

        default_text = (
            SAMPLES[task_name][sample_choice]
            if sample_choice != "Không sử dụng mẫu"
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
            "Nội dung cần phân tích",
            key="input_text",
            height=210,
            placeholder="Nhập nội dung đánh giá phim hoặc tin nhắn tại đây...",
        )

        predict_clicked = st.button(
            "Phân tích văn bản",
            type="primary",
            use_container_width=True,
            disabled=not available_models,
        )

        if predict_clicked:
            text = user_text.strip()
            if not text:
                st.warning("Vui lòng nhập nội dung trước khi phân tích.")
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
                            "result": f"Lỗi: {exc}",
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
                st.error("Mô hình chưa sẵn sàng.")
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
                        "Không thể phân tích văn bản. "
                        "Hãy kiểm tra phiên bản scikit-learn và file mô hình."
                    )
                    st.code(str(exc))

    with right:
        st.subheader("Hướng dẫn sử dụng")
        st.markdown(
            """
            1. Chọn bài toán ở thanh bên.
            2. Chọn một mô hình, hoặc để mặc định `Tất cả` để chạy.
            3. Nhập văn bản hoặc chọn một mẫu thử.
            4. Nhấn **Phân tích văn bản**.
            5. Xem kết quả trong hộp thoại xuất hiện trên màn hình.
            """
        )
        st.divider()
        st.subheader("Mô hình đang chọn")
        if run_all:
            st.write(f"**Tên:** Tất cả mô hình ({len(available_models)})")

            saved = st.session_state.get("all_results")
            has_results = bool(saved) and saved["task"] == task_name
            results_by_model = (
                {r["model"]: r for r in saved["rows"]} if has_results else {}
            )

            summary_rows = []
            for name, artifact in available_models.items():
                metrics = artifact["metrics"]
                row = {"Mô hình": name}
                if has_results:
                    r = results_by_model.get(name)
                    row["Kết quả"] = r["result"] if r else "N/A"
                    row["Độ tin cậy"] = (
                        f"{r['confidence']:.2f}%"
                        if r and r["confidence"] is not None
                        else "N/A"
                    )
                    row["Thời gian (ms)"] = (
                        f"{r['elapsed_ms']:.2f}"
                        if r and r["elapsed_ms"] is not None
                        else "N/A"
                    )
                row["Accuracy"] = format_metric(metrics.get("accuracy"))
                row["F1-score"] = format_metric(metrics.get("f1"))
                summary_rows.append(row)

            if has_results:
                st.caption("Kết quả phân tích gần nhất của tất cả mô hình:")
                with st.expander("Văn bản đã phân tích"):
                    st.write(saved["text"])
            else:
                st.caption(
                    "Chưa có kết quả. Nhấn **Phân tích văn bản** để chạy tất cả mô hình."
                )

            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                hide_index=True,
            )
        elif selected_artifact:
            metrics = selected_artifact["metrics"]
            st.write(f"**Tên:** {model_name}")
            st.write(f"**File:** `{os.path.relpath(selected_artifact['path'], BASE_DIR)}`")
            if metrics:
                st.write(f"**Accuracy:** {format_metric(metrics.get('accuracy'))}")
                st.write(f"**F1-score:** {format_metric(metrics.get('f1'))}")
            else:
                st.caption("File này không kèm metrics benchmark.")

with tab_models:
    st.subheader("Danh sách mô hình đã tìm thấy")
    rows = []

    for current_task, config in TASKS.items():
        task_models = all_models.get(current_task, {})
        for display_name, filename in config["files"].items():
            artifact = task_models.get(display_name)
            metrics = artifact["metrics"] if artifact else {}
            rows.append({
                "Bài toán": current_task,
                "Mô hình": display_name,
                "Trạng thái": "Sẵn sàng" if artifact else "Thiếu file",
                "Accuracy": format_metric(metrics.get("accuracy")),
                "Precision": format_metric(metrics.get("precision")),
                "Recall": format_metric(metrics.get("recall")),
                "F1-score": format_metric(metrics.get("f1")),
                "Tên file": filename,
            })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Các chỉ số được đọc từ metadata lưu trong file mô hình; "
        "đây là kết quả benchmark đã lưu, không phải kết quả tính lại khi chạy ứng dụng."
    )

with tab_about:
    st.subheader("Thông tin hệ thống")
    st.write("Ứng dụng hỗ trợ phân loại văn bản cho hai bài toán: IMDb và SMS Spam.")
    st.write("Các file benchmark được ưu tiên đọc từ thư mục `models/benchmark/`.")
    st.write("Mô hình pipeline xử lý văn bản bằng TF-IDF trước khi phân loại.")
    st.write(f"Số mô hình IMDb đã tải: {len(all_models['IMDb Sentiment Analysis'])}")
    st.write(f"Số mô hình SMS đã tải: {len(all_models['SMS Spam Detection'])}")

    if load_errors:
        st.warning("Một số file mô hình không thể tải.")
        for error in load_errors:
            st.text(error)

    if not all_models["IMDb Sentiment Analysis"] and not all_models["SMS Spam Detection"]:
        st.error(
            "Chưa tải được mô hình nào. Hãy kiểm tra các file .joblib "
            "trong thư mục models/benchmark/."
        )