"""The single orchestrator: ``uv run snake4d <phase> [--env-file FILE] [--set field=value ...]``.

Global context: every user-facing action is a *phase* implemented as ``run(cfg: Config)`` in its own
module.  This file only parses the command line, builds the Config (defaults -> .env -> experiment
env file -> --set) and calls the phase; all real work and all logging happen inside the phases.

Local notes: phases are resolved lazily by import path so that e.g. ``play`` never requires torch
(``logging_utils.versions`` imports it defensively, only to record the CUDA device) and
``train`` never imports pygame (https://docs.python.org/3/library/importlib.html#importlib.import_module).
"""

import argparse
import dataclasses
import importlib
from collections.abc import Callable

from snake4d.config import Config

PHASES: dict[str, str] = {
    "bench": "snake4d.benchmark:run",      # environment/PPO throughput grid -> docs/benchmark.md
    "play": "snake4d.play:run",            # human play in a pygame window
    "train": "snake4d.train:run",          # MaskablePPO training with curriculum + evaluation
    "imitate": "snake4d.imitation:run",    # behaviour-clone the route follower -> bc_model.zip
    "evaluate": "snake4d.evaluation:run",  # scripted policy or saved model -> summary.json
    "report": "snake4d.report:run",        # figures, tables, all_experiments.md, optional PDF
    "publish": "snake4d.publish:run",      # checkpoints + model cards -> Hub repos + collection
    "pipeline": "snake4d.main:pipeline",   # train -> evaluate -> report
}


def resolve(spec: str) -> Callable[[Config], object]:
    """Turn ``"package.module:function"`` into the callable, importing the module on demand."""
    module_name, function_name = spec.split(":")
    return getattr(importlib.import_module(module_name), function_name)


def pipeline(cfg: Config) -> None:
    """Train, then evaluate the model that was just trained (best checkpoint), then report."""
    run_dir = resolve(PHASES["train"])(cfg)
    best = run_dir / "best_model.zip"
    model = best if best.exists() else run_dir / "final_model.zip"
    cfg = dataclasses.replace(cfg, model_path=str(model))  # re-runs the Config guards
    resolve(PHASES["evaluate"])(cfg)
    resolve(PHASES["report"])(cfg)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, build the Config and dispatch to the requested phase.

    Returns ``None`` on purpose: the console script runs ``sys.exit(main())`` and a non-None
    return value would become a failing exit status (https://docs.python.org/3/library/sys.html#sys.exit).
    """
    parser = argparse.ArgumentParser(prog="snake4d", description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--env-file", help="experiment overrides, e.g. experiments/exp02.env")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                        help="override any Config field (repeatable), e.g. --set size=3")
    parser.add_argument("--pdf", action="store_true", help="report: also build reports/paper.pdf")
    args = parser.parse_args(argv)
    cfg = Config.from_env(args.env_file, tuple(args.set))
    phase = resolve(PHASES[args.phase])
    if args.phase == "report" and args.pdf:
        phase(cfg, pdf=True)
    else:
        phase(cfg)


if __name__ == "__main__":  # `uv run python -m snake4d.main train`
    main()
