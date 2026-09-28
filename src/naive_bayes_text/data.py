from __future__ import annotations
import re
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DATA_RAW,
    IMDB_URL,
    SEED,
    SMS_URL,
    TASKS,
    TEST_SIZE,
)

# Cleaning

#: IMDb reviews are glued together with <br />; SMS contains URLs, e-mail
#: addresses and phone numbers. None of them carry class information.
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
URL_RE = re.compile(r"(https?://|www\.)\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
NUM_RE = re.compile(r"\d+")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise one raw document.

    Kept separate from the vectoriser so the UI can reuse it on raw user input
    and so the training path stays free of accidental test-set leakage.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = BR_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = NUM_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


# Acquisition

def prepare_sms(verbose: bool = True) -> Path:
    """Download the UCI archive once and cache ``data/raw/sms_spam.csv``."""
    out = TASKS["sms"]["csv"]
    if out.exists():
        if verbose:
            print(f"[sms]  cached -> {out.name}")
        return out

    archive = DATA_RAW / "sms_spam.zip"
    if not archive.exists():
        if verbose:
            print("[sms]  downloading UCI archive ...")
        urlretrieve(SMS_URL, archive)

    with zipfile.ZipFile(archive) as zf:
        if verbose:
            print(f"[sms]  archive members: {zf.namelist()}")
        # The archive layout has changed over time, so locate the member by name
        # rather than hard-coding "SPAM.txt".
        member = next(
            n for n in zf.namelist()
            if n.lower().replace("_", "").replace(".txt", "").endswith("smsspamcollection")
        )
        blob = zf.read(member).decode("latin-1", errors="replace")

    rows = []
    for line in blob.splitlines():
        if not line.strip():
            continue
        label, _, message = line.partition("\t")
        if label not in TASKS["sms"]["classes"]:
            continue
        rows.append({"label": label, "message": message})

    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    if verbose:
        print(f"[sms]  read {member!r} -> {out.name} ({len(rows)} messages)")
    return out


def prepare_imdb(verbose: bool = True) -> Path:
    """Download aclImdb_v1 once and cache ``data/raw/imdb_reviews.csv``.

    The reviews are read straight out of the tarball, so 33,000 files never
    touch the disk.
    """
    out = TASKS["imdb"]["csv"]
    if out.exists():
        if verbose:
            print(f"[imdb] cached -> {out.name}")
        return out

    archive = DATA_RAW / "aclImdb_v1.tar.gz"
    if not archive.exists():
        if verbose:
            print("[imdb] downloading aclImdb_v1 (84 MB) ...")
        urlretrieve(IMDB_URL, archive)

    rows = []
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            parts = member.name.split("/")
            if len(parts) < 4:
                continue
            split, cls = parts[1], parts[2]
            if split not in ("train", "test") or cls not in ("neg", "pos"):
                continue
            text = tf.extractfile(member).read().decode("utf-8", errors="replace")
            rows.append({"split": split, "label": cls, "text": text})

    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    if verbose:
        print(f"[imdb] wrote {out.name} ({len(rows)} reviews)")
    return out


def prepare_all(verbose: bool = True) -> dict[str, Path]:
    prepare_sms(verbose=verbose)
    prepare_imdb(verbose=verbose)
    return {name: TASKS[name]["csv"] for name in TASKS}


def describe(name: str, verbose: bool = True) -> None:
    """Print one cached dataset's columns, sample rows and class balance.

    Reads only a few rows plus the label column, never the whole file: the IMDb
    CSV is 60+ MB and this function only exists to print three lines of context.
    """
    if not verbose:
        return
    cfg = TASKS[name]
    path = cfg["csv"]
    print(f"\n=== {name} === {path.name} "
          f"({path.stat().st_size / 1024 / 1024:.1f} MB on disk)")
    print(pd.read_csv(path, nrows=3).to_string(max_colwidth=60))
    counts = pd.read_csv(path, usecols=[cfg["label_col"]])[cfg["label_col"]].value_counts()
    print("label counts:\n" + counts.to_string())


# Splitting

def load_task(name: str) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return ``(X_train, X_test, y_train, y_test)`` for one task.

    IMDb ships an official split and is used as-is. SMS is split stratified with
    a fixed seed so the result is reproducible.
    """
    cfg = TASKS[name]
    df = pd.read_csv(cfg["csv"]).dropna(subset=[cfg["text_col"], cfg["label_col"]])

    X = df[cfg["text_col"]].map(clean_text)
    y = df[cfg["label_col"]].astype(str)

    if cfg["use_official_split"] and "split" in df.columns:
        train_mask = (df["split"] == "train").to_numpy()
        test_mask = (df["split"] == "test").to_numpy()
        kind = "official split"
    else:
        idx_train, idx_test = train_test_split(
            np.arange(len(df)), test_size=TEST_SIZE, stratify=y, random_state=SEED
        )
        train_mask = np.zeros(len(df), bool)
        test_mask = np.zeros(len(df), bool)
        train_mask[idx_train] = True
        test_mask[idx_test] = True
        kind = "stratified split"

    print(f"[{name}] {kind} -> train={train_mask.sum()}  test={test_mask.sum()}")
    return (
        X[train_mask].reset_index(drop=True),
        X[test_mask].reset_index(drop=True),
        y[train_mask].reset_index(drop=True),
        y[test_mask].reset_index(drop=True),
    )


def load_all() -> dict[str, tuple]:
    return {name: load_task(name) for name in TASKS}


def make_validation_split(X_train, y_train, seed: int = SEED, val_size: float | None = None):
    """Carve a stratified validation slice out of the training set."""
    from .config import VAL_SIZE

    return train_test_split(
        X_train, y_train,
        test_size=VAL_SIZE if val_size is None else val_size,
        stratify=y_train,
        random_state=seed,
    )
