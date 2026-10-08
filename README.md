# Naive Bayes for Text Classification — CS313

| Task | Train | Test | Features | best α | cut-off | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| SMS (spam) | 4,459 | 1,115 | 11,373 | 0.03 | 0.55 | 0.9848 | 0.9925 | 0.8926 | **0.9399** | 0.9889 |
| IMDb (pos) | 25,000 | 25,000 | 50,000 | 1.0 | 0.45 | 0.8729 | 0.8626 | 0.8872 | **0.8747** | 0.9451 |

Precision/recall/F1 are for the positive class. SMS is 87% ham, so per-class F1 matters
far more than accuracy there.

## Run it

```powershell
pip install -r requirements.txt
jupyter lab notebooks/naive_bayes_text.ipynb     # Run All -> every artefact
```

The notebook is **self-contained** — it imports nothing from `src/` and only needs
`scikit-learn pandas numpy joblib matplotlib`. The first run downloads ~80 MB into
`data/raw/`; after that everything is offline.

Optional, the same experiments as a CLI:

```powershell
pip install -e .
python -m naive_bayes_text all      # train, measure, export
python -m naive_bayes_text verify   # 50 consistency checks
python -m pytest                    # 26 smoke tests
```


## Layout

```
notebooks/naive_bayes.ipynb  the pipeline, self-contained
src/naive_bayes/             the same experiments as an importable library + CLI
tests/                            smoke tests
data/raw/                         cached datasets          (git-ignored)
data/demo_samples.txt             demo sentences + predicted labels
models/                           *_bundle.joblib, *_model.pkl, *_vectorizer.pkl (git-ignored)
reports/                          metrics_table.md, metrics.json, benchmark_table.csv
docs/                             report_template.tex -> report.tex, confusion_matrices.png
app.py                       Streamlit UI source code
```

`src/` and `notebooks/` are two independent implementations of the same experiments and
produce identical numbers, so they can be checked against each other.

## Using the model

```python
import joblib
b     = joblib.load("models/sms_bundle.joblib")
proba = b["model"].predict_proba([text])[0]     # Pipeline: vectorizer + NB
i     = (proba >= b["threshold"]).argmax()      # tuned cut-off, 0.55 sms / 0.45 imdb
label, conf = b["label_map"][i], float(proba[i])
```

`b["task"]` is `"spam"` or `"sentiment"`. Plain `proba.argmax()` is also correct — it just
uses a 0.5 cut-off. Keys: `task`, `model`, `threshold`, `classes`, `label_map`, `metrics`,
`meta`.

# Naive Bayes Text Classification - Mini Project

Ứng dụng minh họa thuật toán **Naive Bayes** trong hai bài toán xử lý ngôn ngữ tự nhiên (NLP):
1. **Phân loại tin nhắn SMS (Spam / Ham)**
2. **Phân tích cảm xúc đánh giá phim IMDb (Positive / Negative)**

---

## Tính năng của Giao diện (UI)

* **Chuyển đổi bài toán dễ dàng:** Sidebar cho phép chọn linh hoạt giữa phân loại SMS Spam và phân tích cảm xúc IMDb.
* **Mẫu dùng thử nhanh:** Tích hợp sẵn các mẫu tin nhắn / đánh giá để kiểm thử tức thì.
* **Hiển thị kết quả dạng Modal Popup:** Nổi đè chính giữa màn hình với thông tin nhãn dự đoán, xác suất chi tiết và thời gian xử lý (ms).
* **Nút làm sạch:** Xóa nhanh nội dung ô nhập liệu.

---

## Hướng dẫn Cài đặt & Khởi chạy
Mở Terminal / PowerShell tại thư mục gốc của dự án và chạy lệnh:
### 1. Cài đặt môi trường & thư viện
```bash
pip install -r requirements.txt
```
### 2.Run
```bash
streamlit run app.py
```