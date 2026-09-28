from __future__ import annotations
import gc
import time
import pandas as pd
from . import data as data_mod
from . import experiments, exporting
from .config import DOCS_DIR, MODELS_DIR, REPORTS_DIR, TASKS


def train_all(verbose: bool = True, core_only: bool = False) -> dict:
    started = time.perf_counter()

    if verbose:
        print("STAGE 0  datasets")
    data_mod.prepare_all(verbose=verbose)
    for name in TASKS:
        data_mod.describe(name, verbose=verbose)
    gc.collect()
    splits = data_mod.load_all()
    gc.collect()

    if verbose:
        print("STAGE 1  main result (alpha -> threshold -> one read of test)")
    main = experiments.run_main(splits, verbose=verbose)
    gc.collect()

    if verbose:
        print("STAGE 1b  untuned baseline and single-call latency")
    untuned = experiments.run_untuned_baseline(splits, verbose=verbose)
    gc.collect()
    latency = experiments.run_latency_split(splits, main["best_alpha"], verbose=verbose)
    gc.collect()

    if verbose:
        print("STAGE 2  benchmark against other classifiers")
    benchmark = experiments.run_benchmark(splits, verbose=verbose)
    benchmark.to_csv(REPORTS_DIR / "benchmark_table.csv", index=False)
    gc.collect()
    if verbose:
        print(f"wrote {REPORTS_DIR / 'benchmark_table.csv'}")

    if core_only:
        if verbose:
            print("\n[core_only] skipping the ablation and improvement search")
        ablation = pd.DataFrame()
        improvements = {"variants": pd.DataFrame(), "calibration": pd.DataFrame(),
                        "thresholds": pd.DataFrame()}
    else:
        if verbose:
            print("STAGE 3  ablation")
        ablation = experiments.run_ablation(splits, main["best_alpha"], verbose=verbose)
        gc.collect()

        if verbose:
            print("STAGE 4  improvement search")
        improvements = experiments.run_improvements(splits, main, verbose=verbose)
        gc.collect()

    if verbose:
        print("STAGE 5  export")
    bundles, plain_sizes = {}, {}
    from .modeling import make_pipeline
    for name, (X_train, X_test, y_train, y_test) in splits.items():
        pipe = make_pipeline(alpha=main["best_alpha"][name])
        pipe.fit(X_train, y_train)
        bundles[name] = exporting.save_bundle(
            name, pipe, main["results"][name], main["best_alpha"][name],
            main["best_threshold"][name], len(X_train), len(X_test), verbose=verbose,
        )
        plain_sizes[name] = exporting.save_plain_pickles(name, pipe, verbose=verbose)

    # A measurement output, not part of training; the report embeds it.
    experiments.plot_confusion_matrices(
        splits, main["y_pred"], path=DOCS_DIR / "confusion_matrices.png")

    exporting.write_demo_samples(bundles, verbose=verbose)
    if core_only:
        if verbose:
            print("[core_only] metrics_table.md / metrics.json left untouched "
                  "(they would lose the ablation and improvement sections)")
    else:
        exporting.write_metrics_report(main, ablation, benchmark, improvements,
                                       bundles, untuned, latency, verbose=verbose)
        exporting.write_metrics_json(main, ablation, benchmark, improvements,
                                     bundles, untuned, latency, verbose=verbose)

    elapsed = time.perf_counter() - started
    if verbose:
        print("\n" + "=" * 70)
        print(f"DONE in {elapsed:.1f} s")
        print("=" * 70)
        for name in TASKS:
            row = main["results"][name]
            print(f"  {name:5s} F1={row['f1']:.4f} acc={row['accuracy']:.4f} "
                  f"ROC-AUC={row['roc_auc']:.4f} alpha={main['best_alpha'][name]:g} "
                  f"threshold={main['best_threshold'][name]:g}")
        print(f"\n  models   -> {MODELS_DIR}")
        print(f"  reports  -> {REPORTS_DIR}")
        print("  report   -> not built here; run "
              "`python -m naive_bayes_text report`")

    return {
        "splits": splits,
        "main": main,
        "ablation": ablation,
        "benchmark": benchmark,
        "improvements": improvements,
        "untuned": untuned,
        "latency": latency,
        "bundles": bundles,
        "plain_pkl_sizes": plain_sizes,
        "elapsed_s": elapsed,
    }
