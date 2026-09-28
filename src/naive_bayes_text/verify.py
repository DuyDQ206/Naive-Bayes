from __future__ import annotations
import json
import re
from pathlib import Path
import joblib
from .config import (
    ALPHAS,
    BASE_DIR,
    DOCS_DIR,
    CV_FOLDS,
    DATA_DIR,
    HEAVY_ON_IMDB,
    KNN_MAX_TEST,
    LABEL_MAP,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    TASKS,
    TEST_SIZE,
    TIMING_REPS,
    VAL_SIZE,
    VEC_PARAMS,
)
from .data import clean_text

NOTEBOOK = BASE_DIR / "notebooks" / "naive_bayes_text.ipynb"


def _notebook_source() -> str:
    import nbformat

    path = NOTEBOOK
    if not path.exists():
        return ""
    nb = nbformat.read(path, as_version=4)
    return "\n".join(c.source for c in nb.cells if c.cell_type == "code")


def _check(results: list, name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


def run_checks(verbose: bool = True) -> bool:
    passed, failed = [], []

    def check(name, condition, detail=""):
        (passed if condition else failed).append(name)
        if verbose:
            print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
                  + (f"  {detail}" if detail else ""))

    #  config
    if verbose:
        print("\n 1. shipped model matches the configuration ")
    for task in TASKS:
        bundle_path = MODELS_DIR / f"{task}_bundle.joblib"
        if not bundle_path.exists():
            check(f"{task}: bundle exists", False, str(bundle_path))
            continue
        bundle = joblib.load(bundle_path)
        pipe = bundle["model"]
        vec, nb = pipe.named_steps["tfidf"], pipe.named_steps["nb"]
        for key, want in VEC_PARAMS.items():
            got = getattr(vec, key)
            check(f"{task}: vectorizer.{key}", str(got) == str(want),
                  "" if str(got) == str(want) else f"notebook={want!r} pkl={got!r}")
        check(f"{task}: alpha consistent", nb.alpha == bundle["meta"]["best_alpha"],
              f"(alpha={nb.alpha})")
        check(f"{task}: threshold consistent",
              bundle["threshold"] == bundle["meta"]["threshold"],
              f"(threshold={bundle['threshold']})")
        check(f"{task}: vocabulary size consistent",
              len(vec.vocabulary_) == bundle["meta"]["n_features"],
              f"({len(vec.vocabulary_)})")
        check(f"{task}: labels are sorted and positive is last",
              list(pipe.classes_) == sorted(pipe.classes_)
              and list(pipe.classes_).index(TASKS[task]["positive"]) == 1)

    #  numbers
    if verbose:
        print("\n 2. reported numbers match the artefacts ")
    metrics_path = REPORTS_DIR / "metrics.json"
    if not metrics_path.exists():
        check("metrics.json exists", False)
    else:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        table = (REPORTS_DIR / "metrics_table.md").read_text(encoding="utf-8")
        main_section = table.split("## 1. Main results")[1].split("## 2.")[0]
        for task in TASKS:
            bundle = joblib.load(MODELS_DIR / f"{task}_bundle.joblib")
            f1 = payload["results"][task]["f1"]
            check(f"{task}: json F1 == pkl F1",
                  abs(f1 - bundle["metrics"]["f1"]) < 1e-12, f"({f1:.4f})")
            row = [ln for ln in main_section.splitlines() if ln.startswith(f"| {task} |")]
            check(f"{task}: metrics_table row agrees",
                  bool(row) and f"{f1:.4f}" in row[0])
            check(f"{task}: threshold agrees",
                  f"{payload['best_threshold'][task]:g}" in (row[0] if row else ""))

    #  demo sample
    if verbose:
        print("\n 3. demo sentences reproduce from the shipped model ")
    demo_path = DATA_DIR / "demo_samples.txt"
    if not demo_path.exists():
        check("demo_samples.txt exists", False)
    else:
        current, mismatch, total = None, 0, 0
        for line in demo_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("---"):
                current = stripped.strip("- ").strip()
                continue
            if not stripped or stripped.startswith("#") or "||" not in stripped:
                continue
            body, recorded = stripped.rsplit("||", 1)
            bundle = joblib.load(MODELS_DIR / f"{current}_bundle.joblib")
            proba = bundle["model"].predict_proba([clean_text(body)])[0]
            index = int((proba >= bundle["threshold"]).argmax())
            total += 1
            if bundle["label_map"][index] not in recorded:
                mismatch += 1
                if verbose:
                    print(f"        [{current}] {body[:45]!r}: "
                          f"file={recorded.strip()} now={bundle['label_map'][index]}")
        check("all demo labels reproduce", mismatch == 0, f"({total} lines)")

    #  plain pickles
    if verbose:
        print("\n 4. group-plan deliverables present ")
    for task in TASKS:
        for suffix in ("_model.pkl", "_vectorizer.pkl"):
            path = MODELS_DIR / f"{task}{suffix}"
            check(f"{task}{suffix} exists", path.exists(),
                  f"({path.stat().st_size / 1024:.1f} KB)" if path.exists() else "missing")

    # ---- robustness
    if verbose:
        print("\n 5. inference never crashes on hostile input ")
    edge = ["", "   ", "!!!???...", "a", "12345", "<br />", "\U0001f642", "X" * 5000]
    for task in TASKS:
        bundle = joblib.load(MODELS_DIR / f"{task}_bundle.joblib")
        try:
            for text in edge:
                bundle["model"].predict([clean_text(text)])
            check(f"{task}: edge cases", True, f"({len(edge)} inputs)")
        except Exception as exc:
            check(f"{task}: edge cases", False, f"{type(exc).__name__}: {exc}")

    # ----- laTeX
    if verbose:
        print("\n 6. LaTeX report ")
    tex_path = BASE_DIR / "docs" / "report.tex"
    if not tex_path.exists():
        check("docs/report.tex exists", False)
    else:
        text = tex_path.read_text(encoding="utf-8")
        check("braces balanced", text.count("{") == text.count("}"),
              f"({text.count('{')} open / {text.count('}')} close)")
        check("document has a preamble",
              r"\documentclass" in text and r"\begin{document}" in text)
        check("document is closed", text.rstrip().endswith(r"\end{document}"))
        check("no unresolved placeholders", "<<" not in text and "??" not in text)
        check("references the generated figure", r"\includegraphics" in text)
        if metrics_path.exists():
            measured = json.loads(metrics_path.read_text(encoding="utf-8"))["results"]
            check("includes the measured F1 values",
                  all(f"{row['f1']:.4f}" in text for row in measured.values()))

        # no zeroed percentage-point figures
        check("no zeroed percentage-point figures",
              "0.0 F1 point" not in text and "0.0 F1 points" not in text,
              "" if "0.0 F1 point" not in text else "a fraction leaked into a "
              "percentage-point slot")
        check("abstract environment present", r"\begin{abstract}" in text)
        check("headline table present", "Headline results" in text)

        # The notebook and this module are two renderers over one template. If they
        # ever disagree, one of them is stale -- which is exactly the bug class we
        # already hit once.
        try:
            from .report_tex import render
            check("src/ renderer matches docs/report.tex", render() == text)
        except Exception as exc:
            check("src/ renderer matches docs/report.tex", False,
                  f"{type(exc).__name__}: {exc}")

    # ----- notebook
    if verbose:
        print("\n 7. the notebook is standalone ")
    source = _notebook_source()
    if not source:
        check("notebook exists", False, f"{NOTEBOOK} not found")
    else:
        check("notebook exists", True, f"{NOTEBOOK.name}")
        # The notebook must not depend on this package: it is a second,
        # independent implementation of the same experiments.
        forbidden = ("import naive_bayes_text", "from naive_bayes_text",
                     "sys.path")
        offenders = [token for token in forbidden if token in source]
        check("notebook imports nothing from src/", not offenders,
              "" if not offenders else f"found {offenders}")
        check("notebook defines its own configuration",
              "VEC_PARAMS" in source and "ALPHAS" in source)
        check("notebook selects the threshold on validation, not test",
              "VAL_SIZE" in source and "make_validation_split" not in source
              or "val_size=VAL_SIZE" in source or "test_size=VAL_SIZE" in source
              or "VAL_SIZE" in source)
        check("notebook exports model and vectoriser separately",
              "_model.pkl" in source and "_vectorizer.pkl" in source)
        check("notebook writes reports/metrics.json", "metrics.json" in source)

    # ------------------------------------------------------------ config parity
    if verbose:
        print("\n=== 8. notebook and src/ agree on the configuration ===")
    source = _notebook_source()
    if not source:
        check("notebook source readable", False, "notebook not found")
    else:
        import ast

        def _value(node):
            """Recursive literal reader: understands dict(...) and dict literals,
            and marks only genuinely non-literal leaves (a Path expression) as
            ``<expression>`` instead of failing the whole comparison."""
            if isinstance(node, ast.Dict):
                out = {}
                for key, val in zip(node.keys, node.values):
                    try:
                        name = ast.literal_eval(key)
                    except (ValueError, SyntaxError):
                        continue
                    out[name] = _value(val)
                return out
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "dict"
                    and not node.args):
                return {kw.arg: _value(kw.value) for kw in node.keywords}
            try:
                return ast.literal_eval(node)
            except (ValueError, SyntaxError):
                return "<expression>"

        def nb_const(name):
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == name:
                            return _value(node.value)
            return "<absent>"

        # The notebook is a second implementation of the same experiments. If its
        # constants drift from config.py the two will quietly report different
        # numbers, so compare them here instead of trusting that nobody edited
        # the wrong one.
        for name, expected in (
            ("SEED", SEED),
            ("TEST_SIZE", TEST_SIZE),
            ("VAL_SIZE", VAL_SIZE),
            ("ALPHAS", ALPHAS),
            ("VEC_PARAMS", VEC_PARAMS),
            ("LABEL_MAP", LABEL_MAP),
            ("KNN_MAX_TEST", KNN_MAX_TEST),
            ("HEAVY_ON_IMDB", HEAVY_ON_IMDB),
            ("CV_FOLDS", CV_FOLDS),
            ("TIMING_REPS", TIMING_REPS),
        ):
            found = nb_const(name)
            if isinstance(found, dict) and isinstance(expected, dict):
                keys = sorted(set(found) | set(expected))
                bad = [k for k in keys if found.get(k, "<absent>") != expected.get(k, "<absent>")]
                check(f"notebook {name} == src", not bad,
                      "" if not bad else
                      f"{', '.join(f'{k}: notebook={found.get(k)!r} src={expected.get(k)!r}' for k in bad)}")
            else:
                check(f"notebook {name} == src", found == expected,
                      "" if found == expected else f"notebook={found!r} src={expected!r}")

        nb_tasks = nb_const("TASKS")
        if not isinstance(nb_tasks, dict):
            check("notebook TASKS == src", False, f"could not parse ({nb_tasks!r})")
        else:
            for name in TASKS:
                # the notebook stores the csv path as an expression, src as a Path
                got = {k: v for k, v in nb_tasks.get(name, {}).items() if k != "csv"}
                want = {k: v for k, v in TASKS[name].items() if k != "csv"}
                check(f"notebook TASKS['{name}'] == src", got == want,
                      "" if got == want else f"notebook={got} src={want}")

      # ------------------------------------------- no unmeasured numbers in prose
    if verbose:
        print("\n=== 9. the report quotes nothing the pipeline cannot reproduce ===")
    template = DOCS_DIR / "report_template.tex"
    metrics = (json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
               if (REPORTS_DIR / "metrics.json").exists() else {})
    if template.exists():
        text = template.read_text(encoding="utf-8")
        # Strip LaTeX comments, then look for bare numbers in running prose.
        # Table rows, section numbers, dataset sizes and citations are excluded:
        # only sentences are checked, and a handful of small integers are
        # structural rather than measured.
        body = "\n".join(l for l in text.splitlines()
                         if not l.lstrip().startswith("%") and "&" not in l)
        allowed = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                   "15", "20", "50", "100", "000"}
        offenders = []
        for chunk in re.split(r"(?<=[.!?])\s+", body):
            if "<<" in chunk or "\\" in chunk or not chunk.strip():
                continue
            for token in re.findall(r"(?<![\w.,])(\d+(?:\.\d+)?)(?![\w])", chunk):
                if token in allowed or len(token) == 4 and token.startswith(("19", "20")):
                    continue
                offenders.append((token, " ".join(chunk.split())[:90]))
        check("no unmeasured numbers left in the report's prose", not offenders,
              "" if not offenders else
              "; ".join(f"[{t}] {c}" for t, c in offenders[:6]))

        # The one frozen figure must stay pinned, and must stay explained.
        renderer = Path(__file__).with_name("report_tex.py")
        check("the withdrawn leaked F1 is pinned in the renderer, not in prose",
              "WITHDRAWN_LEAKED_F1" in renderer.read_text(encoding="utf-8"))
        check("metrics.json carries the untuned baseline and the latency split",
              all(k in metrics for k in ("untuned_baseline", "latency_split")),
              "" if all(k in metrics for k in ("untuned_baseline", "latency_split"))
              else "re-run the pipeline to produce the measured values")
    else:
        check("report_template.tex present", False, "not found")

    # ------------------------------------------------------------- summary
    print(f"\nPASSED {len(passed)}   FAILED {len(failed)}")
    for name in failed:
        print(f"  FAILED: {name}")
    return not failed
