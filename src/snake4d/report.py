"""Report phase: figures and tables from run artifacts, the cross-experiment summary, and the PDF.

Global context: every experiment's prose lives once in ``reports/experiments/expNN_<name>.md``.
This module generates what can be generated - learning-curve figures per training run, a
cross-run metrics table (``reports/all_experiments.md``), copies of the small run artifacts
(``reports/data/<run>/``) - and, on request, concatenates the markdown into ``reports/paper.pdf``.

Local notes:
* SB3's ``progress.csv`` (logger format ``csv``) and ``evaluations.npz`` (``EvalCallback``:
  ``timesteps``, ``results``, ``ep_lengths``, ``successes``) are read with pandas/numpy -
  https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
* ``stable_baselines3.common.monitor.load_results`` reads VecMonitor's ``*.monitor.csv``.
* Figures: matplotlib ``subplots`` (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html).
* PDF: ``markdown_pdf`` (AGPL, docs dependency group only, imported lazily) -
  https://pypi.org/project/markdown-pdf/
"""

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import load_results

from snake4d.config import Config
from snake4d.logging_utils import make_run_dir, setup_logging

REPORTS = Path("reports")
DATA_FILES = ("summary.json", "config.json", "versions.json", "eval_episodes.csv",
              "benchmark.json", "eval/evaluations.npz", "progress.csv", "run.log")
log = logging.getLogger("snake4d.report")


def run_name(run_dir: Path) -> str:
    """``20260904-190000_train_exp02`` -> ``exp02`` (the experiment/run label after the phase)."""
    return run_dir.name.split("_", 2)[2]


def plot_run(run_dir: Path, out_dir: Path) -> list[Path]:
    """Learning curves (fill, eval success, curriculum frontier, episode length), fill histogram."""
    import matplotlib

    matplotlib.use("Agg")  # headless backend for scripts/tests
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    progress = read_progress(run_dir)
    if progress is not None:
        x = progress["time/total_timesteps"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        panels = [
            (("rollout/fill_mean", "rollout/fill_mean_true_start"), "fill fraction (rollouts)"),
            (("eval/success_rate",), "eval success rate (true start)"),
            (("curriculum/frontier",), "curriculum frontier (start length)"),
            (("rollout/ep_len_mean",), "episode length"),
        ]
        for ax, (columns, title) in zip(axes.flat, panels, strict=True):
            for column in columns:
                if column in progress:
                    series = progress[column]
                    ax.plot(x[series.notna()], series.dropna(), label=column.split("/")[-1])
            ax.set_title(title)
            ax.set_xlabel("timesteps")
            ax.legend(fontsize=8)
        fig.suptitle(run_name(run_dir))
        fig.tight_layout()
        path = out_dir / f"{run_name(run_dir)}_curves.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    if any(run_dir.glob("*.monitor.csv")):
        episodes = load_results(str(run_dir))  # VecMonitor rows: r, l, t, is_success, fill, ...
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(episodes["fill"], bins=20, range=(0, 1))
        ax.set_title(f"{run_name(run_dir)}: terminal fill of {len(episodes)} training episodes")
        ax.set_xlabel("fill at episode end")
        fig.tight_layout()
        path = out_dir / f"{run_name(run_dir)}_fill_hist.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


def read_progress(run_dir: Path) -> pd.DataFrame | None:
    """SB3's progress.csv, or None when it is missing/empty (a run that has just started)."""
    path = run_dir / "progress.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def last_value(progress: pd.DataFrame, column: str):
    """Last non-missing value of a column (eval-only dump rows leave gaps), or None."""
    if column not in progress:
        return None
    series = progress[column].dropna()
    return series.iloc[-1] if len(series) else None


def summarize_run(run_dir: Path) -> dict:
    """One table row per run: board, phase, headline metrics read from its artifacts."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    row = {"run": run_name(run_dir), "phase": run_dir.name.split("_")[1],
           "board": f"{config['size']}^{config['ndim']}", "run_dir": run_dir.name}
    progress = read_progress(run_dir)
    if progress is not None:
        for column, key in (("time/total_timesteps", "timesteps"), ("time/fps", "fps")):
            value = last_value(progress, column)
            if value is not None:
                row[key] = int(value)
        for column, key, scale in (("time/time_elapsed", "wall_min", 1 / 60),
                                   ("rollout/fill_mean", "fill_final", 1.0),
                                   ("rollout/fill_mean_true_start", "fill_true_start_final", 1.0)):
            value = last_value(progress, column)
            if value is not None:
                row[key] = round(float(value) * scale, 3)
    evaluations = run_dir / "eval" / "evaluations.npz"
    if evaluations.exists():
        data = np.load(evaluations)
        if "successes" in data.files:
            rates = data["successes"].mean(axis=1)
            row["eval_success_final"] = round(float(rates[-1]), 3)
            row["eval_success_best"] = round(float(rates.max()), 3)
            row["eval_success_best_at"] = int(data["timesteps"][rates.argmax()])
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for mode, stats in summary["modes"].items():
            tag = "det" if mode.endswith("True") else "stoch"
            row[f"success_{tag}"] = (f"{stats['success_rate_mean']:.3f}"
                                     f" +- {stats['success_rate_std']:.3f}")
            row[f"fill_{tag}"] = round(stats["fill_mean"], 3)
            row[f"steps_to_complete_{tag}"] = stats["steps_to_complete_mean"]
    return row


def episode_stats(run_dir: Path) -> dict | None:
    """Per-run episode statistics from VecMonitor's CSV, split by true vs curriculum starts."""
    if not any(run_dir.glob("*.monitor.csv")):
        return None
    episodes = load_results(str(run_dir))
    true_start = episodes[episodes["start_len"] == 1]
    curriculum = episodes[episodes["start_len"] > 1]

    def stats(subset: pd.DataFrame, prefix: str) -> dict:
        """Count, fill mean/max, mean length and success rate of one subset of episodes."""
        if not len(subset):
            return {f"{prefix}_episodes": 0}
        return {
            f"{prefix}_episodes": int(len(subset)),
            f"{prefix}_fill_mean": round(float(subset["fill"].mean()), 4),
            f"{prefix}_fill_max": float(subset["fill"].max()),
            f"{prefix}_len_mean": round(float(subset["l"].mean()), 1),
            f"{prefix}_success_rate": round(float(subset["is_success"].astype(float).mean()), 4),
        }

    return {"episodes": int(len(episodes)), **stats(true_start, "true_start"),
            **stats(curriculum, "curriculum")}


def copy_data(run_dir: Path, data_dir: Path) -> None:
    """Keep the small artifacts of a run under version control (runs/ itself is ignored)."""
    target = data_dir / run_name(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, target / source.name)
    stats = episode_stats(run_dir)
    if stats:  # the monitor CSV itself is large; its summary is what the reports cite
        (target / "episodes.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def write_all_experiments(rows: list[dict], reports_dir: Path) -> Path:
    """``reports/all_experiments.md``: cross-run table plus links; prose stays in the exp files."""
    table = pd.DataFrame(rows)
    for column in ("timesteps", "fps", "eval_success_best_at"):
        if column in table:  # integers stay readable next to missing values (no float cast)
            table[column] = table[column].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
    table = table.fillna("")
    columns = [c for c in ("run", "phase", "board", "timesteps", "wall_min", "fps", "fill_final",
                           "fill_true_start_final", "eval_success_final", "eval_success_best",
                           "eval_success_best_at", "success_det", "success_stoch", "fill_det",
                           "steps_to_complete_det", "run_dir") if c in table]
    experiments = sorted((reports_dir / "experiments").glob("exp*.md"))
    lines = [
        "# All experiments", "",
        "Generated by `uv run snake4d report` from the artifacts in `runs/` (small copies in "
        "`reports/data/`). Each experiment's question, hypothesis, analysis and learnings are in "
        "its own file:", "",
        *[f"- [{p.stem}](experiments/{p.name})" for p in experiments], "",
        "## Metrics by run", "",
        table[columns].to_markdown(index=False) if columns else "(no runs found)", "",
        "Figures: `reports/figures/<run>_curves.png` and `<run>_fill_hist.png` per training run.",
        "",
    ]
    path = reports_dir / "all_experiments.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def to_pdf(reports_dir: Path, pdf_path: Path) -> Path:
    """Concatenate paper.md, all_experiments.md and the experiment files into one PDF."""
    from markdown_pdf import MarkdownPdf, Section  # AGPL tool: docs group, never a runtime import

    parts = [reports_dir / "paper.md", reports_dir / "all_experiments.md",
             *sorted((reports_dir / "experiments").glob("exp*.md"))]
    pdf = MarkdownPdf(toc_level=2, optimize=True)
    for part in parts:
        if part.exists():
            text = part.read_text(encoding="utf-8").replace("](../../", "](").replace("](../", "](")
            pdf.add_section(Section(text, root=str(reports_dir)))
    pdf.meta["title"] = "4D Snake with MaskablePPO"
    pdf.save(str(pdf_path))
    return pdf_path


def run(cfg: Config, pdf: bool = False, reports_dir: Path = REPORTS) -> Path:
    """Phase entry point: figures for every training run, data copies, the table, optional PDF."""
    run_dir = make_run_dir(cfg, "report")
    logger = setup_logging(run_dir)
    runs = sorted(p for p in Path(cfg.runs_dir).iterdir()
                  if p.is_dir() and (p / "config.json").exists() and "_report_" not in p.name)
    rows = []
    for source in runs:
        try:
            if source.name.split("_")[1] == "train":
                for figure in plot_run(source, reports_dir / "figures"):
                    logger.info("figure %s", figure)
            copy_data(source, reports_dir / "data")
            rows.append(summarize_run(source))
        except Exception:  # noqa: BLE001 - one broken/in-progress run must not stop the report
            logger.exception("skipping %s", source)
    table = write_all_experiments(rows, reports_dir)
    logger.info("wrote %s with %d runs", table, len(rows))
    if pdf:
        logger.info("wrote %s", to_pdf(reports_dir, reports_dir / "paper.pdf"))
    return table
