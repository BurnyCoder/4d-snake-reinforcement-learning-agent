"""Timestamped logging and run directories.

Global context: every phase (bench, train, evaluate, report) creates one run directory
``runs/<YYYYmmdd-HHMMSS>_<phase>_<run_name>/`` holding ``run.log`` (everything logged, with
timestamps; SB3's own ``log.txt``/``progress.csv`` live next to it), ``config.json`` (resolved
Config) and ``versions.json`` (library versions, GPU, git commit) so any number in a report can be
traced back to exact code and settings.

Local notes: ``logging.basicConfig(force=True, handlers=[...])`` per
https://docs.python.org/3/library/logging.html#logging.basicConfig - ``force`` removes handlers that
imported libraries (SB3) may already have attached; ``encoding`` must be set on the FileHandler.
"""

import json
import logging
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from snake4d.config import Config

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # canonical format from the
# Python logging HOWTO: https://docs.python.org/3/howto/logging.html#displaying-the-date-time-in-messages
TRACKED_PACKAGES = ("torch", "gymnasium", "stable-baselines3", "sb3-contrib", "numpy", "pygame-ce")


def setup_logging(run_dir: Path) -> logging.Logger:
    """Send INFO logs to both ``run_dir/run.log`` and the terminal, timestamped."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        force=True,
        handlers=[
            logging.FileHandler(Path(run_dir) / "run.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("snake4d")


def versions() -> dict[str, str]:
    """Library versions, CUDA device and git commit for ``versions.json``."""
    info: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            info[name] = version(name)  # https://docs.python.org/3/library/importlib.metadata.html
        except PackageNotFoundError:
            info[name] = "missing"
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        info["git_commit"] = git.stdout.strip() if git.returncode == 0 else "unknown"
    except OSError:  # git not installed / not on PATH must not stop a run
        info["git_commit"] = "unknown"
    try:
        import torch  # imported lazily so phases without torch (play) stay light

        info["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
    except ImportError:
        info["cuda_device"] = "torch missing"
    return info


def make_run_dir(cfg: Config, phase: str) -> Path:
    """Create ``runs/<timestamp>_<phase>_<run_name>`` and persist config + versions inside it."""
    stem = Path(cfg.runs_dir) / f"{datetime.now():%Y%m%d-%H%M%S}_{phase}_{cfg.run_name}"
    run_dir = stem
    for suffix in range(1, 100):  # two runs in the same second must not share a directory
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            run_dir = stem.with_name(f"{stem.name}-{suffix}")
    cfg.to_json(run_dir / "config.json")
    (run_dir / "versions.json").write_text(json.dumps(versions(), indent=2), encoding="utf-8")
    return run_dir
