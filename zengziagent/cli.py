"""``zengzi`` command line entry point (thin dispatcher)."""
from __future__ import annotations

import sys

COMMANDS = {
    "annotate": "zengziagent.experiments.run_annotation",
    "ablate": "zengziagent.experiments.run_ablation",
    "reflect": "zengziagent.experiments.run_reflection",
    "evaluate": "zengziagent.experiments.evaluate",
    "analyze": "zengziagent.experiments.analyze_ablation",
    "tables": "zengziagent.experiments.make_tables",
    "figures": "zengziagent.experiments.make_figures",
    "audit": "zengziagent.evaluation.audit",
    "baseline": "zengziagent.baselines.transformers_baseline",
    "scientometrics": "zengziagent.scientometrics.analysis",
}


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: zengzi <command> [args]\ncommands: " + ", ".join(COMMANDS))
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        sys.exit(f"unknown command '{cmd}'; known: {', '.join(COMMANDS)}")
    import importlib

    mod = importlib.import_module(COMMANDS[cmd])
    result = mod.main(rest)
    if isinstance(result, int):
        sys.exit(result)


if __name__ == "__main__":  # pragma: no cover
    main()
