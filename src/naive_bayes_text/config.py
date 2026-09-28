from __future__ import annotations

from pathlib import Path


# Paths

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

for _p in (DATA_DIR, DATA_RAW, MODELS_DIR, REPORTS_DIR, DOCS_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# Reproducibility

SEED = 42
TEST_SIZE = 0.20
#: Carved out of TRAIN only. Used to choose the decision threshold so that the
#: test set is never used for model selection.
VAL_SIZE = 0.15


# Dataset sources (both public, both English, both binary, no login required)

SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

#: Human-readable provenance, printed in the report and the README.
DATASET_CITATIONS = {
    "sms": (
        "SMS Spam Collection",
        "Almeida, A. A., Gomez Hidalgo, J. M., Yamakami, A. (2011). "
        "Contributions to the Experimental Study of the Spam Filtering Problem. "
        "UCI Machine Learning Repository, dataset id 228.",
    ),
    "imdb": (
        "Large Movie Review Dataset (IMDb)",
        "Maas, A. L., Daly, R., Pham, T., Huang, D., & Potts, C. (2011). "
        "Learning Word Vectors for Sentiment Analysis. ACL-HLT 2011, pp. 142--150.",
    ),
}

TASKS = {
    "sms": {
        "csv": DATA_RAW / "sms_spam.csv",
        "label_col": "label",
        "text_col": "message",
        "classes": ["ham", "spam"],
        "positive": "spam",
        "use_official_split": False,
        "short": "SMS Spam",
    },
    "imdb": {
        "csv": DATA_RAW / "imdb_reviews.csv",
        "label_col": "label",
        "text_col": "text",
        "classes": ["neg", "pos"],
        "positive": "pos",
        "use_official_split": True,
        "short": "IMDb Sentiment",
    },
}

#: Pretty display names used by the UI and by the report.
LABEL_MAP = {
    "sms": {0: "Ham (legit message)", 1: "Spam"},
    "imdb": {0: "Negative", 1: "Positive"},
}


# Text vectorisation

#: TF-IDF settings. Justification for each line is in the notebook and report.
VEC_PARAMS = dict(
    lowercase=True,        # "Free" and "free" are the same signal
    stop_words=None,       # never drop "not"/"no"/"never": flips sentiment
    ngram_range=(1, 2),    # bigrams keep "not good" together
    min_df=2,              # drop hapax legomena -> noise, and shrink the model
    max_df=0.9,            # drop words in >90% of docs -> no class information
    max_features=50_000,   # bound memory and latency
    sublinear_tf=True,     # 1+log(tf): repetition is not proportionally stronger
    token_pattern=r"(?u)\b\w\w+\b",
)

#: Tree ensembles densify the matrix, so the benchmark uses a smaller vocabulary.
BENCH_VEC_PARAMS = dict(VEC_PARAMS, max_features=20_000)


# Model

#: Smoothing grid. Deliberately wide: the SMS optimum (0.03) and the IMDb
#: optimum (1.0) are two orders of magnitude apart.
ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
THRESHOLD_GRID = [round(0.20 + 0.025 * i, 3) for i in range(17)]  # 0.200 .. 0.600

#: Benchmark grid: tree/KNN models are too slow or too large on IMDb.
HEAVY_ON_IMDB = {"KNN (k=5)", "RandomForest(50)"}
KNN_MAX_TEST = 3000
TIMING_REPS = 200
CV_FOLDS = 5

#: Improvement-search variants, ranked on validation and reported on test.
#: ``is_shipped`` is a machine-readable flag: downstream reports must never parse
#: the human-readable label to find the default configuration. The label strings
#: are kept byte-identical to the ones the standalone notebook writes, so both
#: implementations render the same report.
VARIANTS = [
    ("(1,2)gram, max_features=50k  [SHIPPED]", {}, True),
    ("(1,2)gram, max_features=100k", dict(max_features=100_000), False),
    ("(1,2)gram, no vocabulary cap", dict(max_features=None), False),
    ("(1,3)gram, max_features=100k", dict(ngram_range=(1, 3), max_features=100_000), False),
    ("min_df=1,   (1,2)gram, 100k", dict(min_df=1, max_features=100_000), False),
]
