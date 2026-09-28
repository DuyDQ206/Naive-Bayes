from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import BASE_DIR, MODELS_DIR, REPORTS_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m naive_bayes_text",
        description="Train and evaluate Naive Bayes text classifiers for CS313.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("all", help="train, measure and export; the LaTeX report is a separate step")

    p_data = sub.add_parser("data", help="download and cache the two datasets only")
    p_data.set_defaults(stage="data")

    p_report = sub.add_parser("report", help="re-render docs/report.tex from metrics.json")
    p_report.set_defaults(stage="report")

    p_verify = sub.add_parser("verify", help="check that artefacts and reports agree")
    p_verify.set_defaults(stage="verify")

    parser.add_argument("-q", "--quiet", action="store_true", help="less output")
    parser.add_argument("--core", action="store_true",
                        help="skip the ablation and improvement-search stages")
    parser.add_argument("--version", action="version", version=__version__)

    args = parser.parse_args(argv)
    stage = getattr(args, "stage", "all")
    verbose = not args.quiet

    if stage == "data":
        from .config import TASKS
        from .data import describe, load_all, prepare_all
        prepare_all(verbose=verbose)
        for name in TASKS:
            describe(name, verbose=verbose)
        load_all()
        return 0

    if stage == "report":
        from .report_tex import write_tex_report
        print(write_tex_report(verbose=verbose))
        return 0

    if stage == "verify":
        from .verify import run_checks
        return 0 if run_checks(verbose=verbose) else 1

    from .pipeline import train_all
    summary = train_all(verbose=verbose, core_only=args.core)
    print(f"\nArtefacts written under {BASE_DIR}")
    print(f"  {MODELS_DIR.name}/   models for the UI (TV4)")
    print(f"  {REPORTS_DIR.name}/  metrics tables (TV2)")
    print("  docs/report.tex is a separate step: "
          "`python -m naive_bayes_text report`")
    return 0 if summary["main"]["results"] else 1


if __name__ == "__main__":
    sys.exit(main())
