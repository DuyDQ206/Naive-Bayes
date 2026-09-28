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
