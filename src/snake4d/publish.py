"""Publish phase: evaluated checkpoints, configs and evaluation artifacts to the Hugging Face Hub.

Global context: ``train`` / ``imitate`` save ``best_model.zip`` / ``bc_model.zip`` under ``runs/``
and ``evaluate`` writes the ``summary.json`` every report quotes.  This phase packages one Hub model
repo per run (the evaluated checkpoint, ``config.json``, ``versions.json``, the evaluation files,
the learning-curve figures and a model card) and groups the repos in one Hub collection, so the
weights behind ``reports/networks.md`` are downloadable and their provenance is recorded.

Local notes:
* Uploads go through ``huggingface_hub.HfApi``: ``create_repo(exist_ok=True)`` and
  ``upload_folder`` (https://huggingface.co/docs/huggingface_hub/guides/upload),
  ``list_collections`` / ``create_collection`` / ``add_collection_item(exists_ok=True)``
  (https://huggingface.co/docs/huggingface_hub/guides/collections).  The library lives in the
  optional ``hub`` dependency group, so it is imported lazily (like ``markdown_pdf``).
* Authentication is the cached ``hf auth login`` token or the ``HF_TOKEN`` variable
  (https://huggingface.co/docs/huggingface_hub/quick-start#authentication); nothing is read from
  ``.env``.
* The model card copies the Hub's Stable-Baselines3 cards (``library_name: stable-baselines3``,
  ``model-index`` with a ``reinforcement-learning`` task and ``value +/- std`` metrics,
  https://huggingface.co/sb3/ppo-CartPole-v1/blob/main/README.md) and the metadata spec
  (``license``, ``pipeline_tag``, ``base_model``: https://huggingface.co/docs/hub/model-cards).
* Re-running is safe: repos are updated in place, the collection is found by title, items are
  added with ``exists_ok`` so nothing is duplicated.
"""

import json
import logging
import os
import re
import shutil
from pathlib import Path

from snake4d.config import Config
from snake4d.logging_utils import make_run_dir, setup_logging
from snake4d.report import REPORTS, run_name

GITHUB = "https://github.com/BurnyCoder/4d-snake-reinforcement-learning-agent"
HUB = "https://huggingface.co"  # repo pages live at HUB/<namespace>/<repo>
MODEL_FILES = ("best_model.zip", "bc_model.zip", "final_model.zip")  # evaluated checkpoint first
NOTE_LIMIT = 500  # add_collection_item caps a note at 500 characters (collections guide)
DESCRIPTION_LIMIT = 150  # the Hub rejects longer collection descriptions ("Too big", HTTP 400)
COLLECTION_DESCRIPTION = ("MaskablePPO and behaviour-cloned 4D snake networks (2^4, 3^4, 4^4 "
                          "boards) with configs, evaluation files and cards; negative results "
                          "included.")
log = logging.getLogger("snake4d.publish")


# --- locating a run's artifacts ----------------------------------------------------------------
def _newest(runs_dir: Path, pattern: str) -> Path | None:
    """Newest run directory whose name matches ``pattern`` (timestamps sort chronologically)."""
    hits = sorted(p for p in Path(runs_dir).iterdir() if p.is_dir() and re.search(pattern, p.name))
    return hits[-1] if hits else None


def find_run(runs_dir: Path, name: str) -> Path | None:
    """The ``train`` or ``imitate`` run that produced the checkpoint for ``name``."""
    return _newest(runs_dir, rf"_(train|imitate)_{re.escape(name)}(-\d+)?$")


def find_eval(runs_dir: Path, name: str) -> Path | None:
    """The archived evaluation of the checkpoint: ``<name>_best`` (train) or ``_eval`` (imitate)."""
    found = _newest(runs_dir, rf"_evaluate_{re.escape(name)}_(best|eval)(-\d+)?$")
    return found if found is not None and (found / "summary.json").exists() else None


def model_file(run_dir: Path) -> Path:
    """The evaluated checkpoint: best_model.zip, else the clone's bc_model.zip, else final."""
    for candidate in MODEL_FILES:
        if (run_dir / candidate).exists():
            return run_dir / candidate
    raise FileNotFoundError(f"no model zip in {run_dir}")


def repo_name(name: str) -> str:
    """Hub repo name of a run: ``exp02d_ppo_2x4_backplay_strict`` -> ``4d-snake-exp02d-ppo-...``."""
    return "4d-snake-" + name.replace("_", "-")


def write_up(name: str, reports_dir: Path = REPORTS) -> str | None:
    """GitHub URL of the experiment write-up whose file name starts with the run's ``expNN``."""
    prefix = re.match(r"exp\d+", name)
    hits = sorted((reports_dir / "experiments").glob(f"{prefix.group(0)}_*.md")) if prefix else []
    return f"{GITHUB}/blob/main/reports/experiments/{hits[0].name}" if hits else None


def base_run(run_cfg: dict) -> str | None:
    """The run a fine-tune started from (its config's ``model_path``), e.g. ``exp05_bc_4x4``."""
    pattern = r"_(?:train|imitate)_([A-Za-z0-9_]+?)(?:-\d+)?[\\/][^\\/]+\.zip$"
    found = re.search(pattern, run_cfg.get("model_path", ""))
    return found.group(1) if found else None


# --- the model card ----------------------------------------------------------------------------
def describe(phase: str, run_cfg: dict) -> str:
    """One clause naming a run's learning method and curriculum, read from its config."""
    if phase == "imitate":
        return "behaviour cloning of the Hamiltonian route follower (`snake4d imitate`)"
    if run_cfg.get("model_path"):
        method = "MaskablePPO fine-tuned from the behaviour-cloned network"
    else:
        method = "MaskablePPO trained from scratch"
    if run_cfg.get("curriculum"):
        curriculum = (f"Backplay reverse curriculum (gate {run_cfg['curriculum_rho']}, "
                      f"window {run_cfg['curriculum_window']})")
    else:
        curriculum = "no curriculum"
    return f"{method}, {curriculum}, {run_cfg['total_timesteps']:,} environment steps"


def _row(label: str, mode: dict) -> str:
    """One results-table row from a ``summary.json`` mode block."""
    steps = mode["steps_to_complete_mean"]
    steps_text = "never" if steps is None else f"{steps:,.1f}"
    return (f"| {label} | {mode['success_rate_mean']:.3f} +- {mode['success_rate_std']:.3f} | "
            f"{mode['fill_mean']:.3f} | {steps_text} | {mode['win_within']['4C']:.3f} |")


def _metadata(repo: str, board_tag: str, det: dict, tags: list[str],
              base_model: str | None) -> list[str]:
    """The YAML front matter: licence, library, tags and the ``model-index`` evaluation results."""
    metrics = [("completion_rate", "completion rate (deterministic, 100 episodes x 3 seeds)",
                f"{det['success_rate_mean']:.3f} +/- {det['success_rate_std']:.3f}"),
               ("mean_reward", "mean episode return (deterministic)", f"{det['return_mean']:.2f}"),
               ("fill", "mean final fill (deterministic)", f"{det['fill_mean']:.3f}")]
    if det["steps_to_complete_mean"] is not None:
        metrics.append(("steps_to_complete", "mean steps to complete (deterministic, won episodes)",
                        f"{det['steps_to_complete_mean']:.1f}"))
    lines = ["---", "license: mit", "library_name: stable-baselines3",
             "pipeline_tag: reinforcement-learning", "tags:", *[f"- {tag}" for tag in tags]]
    lines += [f"base_model: {base_model}"] if base_model else []
    lines += ["model-index:", f"- name: {repo}", "  results:", "  - task:",
              "      type: reinforcement-learning", "      name: reinforcement-learning",
              "    dataset:", f"      name: {board_tag}", f"      type: {board_tag}",
              "    metrics:"]
    for kind, label, value in metrics:
        lines += [f"    - type: {kind}", f"      name: {label}", f"      value: {value}",
                  "      verified: false"]
    return lines + ["---", ""]


def model_card(name: str, phase: str, run_cfg: dict, summary: dict, versions: dict, repo_id: str,
               zip_name: str, base_model: str | None = None, write_up_url: str | None = None,
               collection_url: str | None = None, figure_names: tuple[str, ...] = ()) -> str:
    """README.md of one Hub repo: SB3-style metadata block plus a plain-words body (ASCII only)."""
    size, ndim = run_cfg["size"], run_cfg["ndim"]
    board_tag, cells = f"4d-snake-{size}x{ndim}", size**ndim
    det = summary["modes"]["deterministic=True"]
    stoch = summary["modes"].get("deterministic=False")
    repo = repo_id.split("/")[-1]
    tags = ["4d-snake", board_tag, "snake", "reinforcement-learning", "deep-reinforcement-learning",
            "stable-baselines3", "sb3-contrib", "maskable-ppo"]
    tags += ["behavior-cloning", "imitation-learning"] if phase == "imitate" else []
    tags += ["backplay-curriculum"] if phase == "train" and run_cfg.get("curriculum") else []
    if det["success_rate_mean"] == 1.0:
        outcome = "completes the board in every deterministic evaluation episode"
    else:
        outcome = (f"completes the board in {100 * det['success_rate_mean']:.1f} % of "
                   "deterministic episodes")
    docs = f"{GITHUB}/blob/main/docs"
    body = [f"# {repo}", "",
            f"A `{run_cfg['net_width']}x{run_cfg['net_width']}` MLP policy for "
            f"**{ndim}-dimensional snake on the {size}^{ndim} board** ({cells} cells, "
            f"{2 * ndim} moves): "
            f"{describe(phase, run_cfg)}. From a length-1 start it {outcome}, evaluated with the "
            f"protocol of [docs/evaluation.md]({docs}/evaluation.md) (100 episodes x 3 seeds, "
            "masked `evaluate_policy`).", ""]
    if det["success_rate_mean"] == 0.0:
        body += ["**Negative result.** This network never fills the board from the true start "
                 f"(mean final fill {det['fill_mean']:.3f}). It is published so the failure is "
                 "reproducible; the write-up linked below analyses why.", ""]
    body += ["## Results (`eval/summary.json`)", "",
             "| mode | completion +- std | mean fill | steps to complete | won within 4C |",
             "|---|---|---|---|---|", _row("deterministic (argmax)", det)]
    body += [_row("sampling", stoch)] if stoch else []
    body += ["", "## How to use", "",
             "The observation is this repository's `4*C + 2` float vector and the action space its "
             f"`2*ndim` masked moves ([docs/game_rules.md]({docs}/game_rules.md)), so the "
             "checkpoint runs inside `snake4d`'s environment:", "", "```bash",
             f"git clone {GITHUB}.git && cd {GITHUB.rsplit('/', 1)[-1]} && uv sync",
             f"hf download {repo_id} {zip_name} --local-dir weights",
             f"uv run snake4d evaluate --set model_path=weights/{zip_name} --set size={size} "
             f"--set ndim={ndim}",
             "```", "", "```python",
             "# https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html",
             "from sb3_contrib import MaskablePPO",
             "from sb3_contrib.common.maskable.utils import get_action_masks",
             "from snake4d.config import Config", "from snake4d.vec_env import make_env", "",
             f"cfg = Config(size={size}, ndim={ndim})",
             f'model = MaskablePPO.load("weights/{zip_name}", device="cpu")',
             "env = make_env(cfg, 1, 0)  # one board; observation shape (1, 4*C + 2)",
             "obs = env.reset()",
             "masks = get_action_masks(env)  # the legal moves, one row per board",
             "action, _ = model.predict(obs, action_masks=masks, deterministic=True)",
             "```", ""]
    if phase == "imitate" or base_model:
        body += ["Use deterministic mode: the cloned policy follows a fixed Hamiltonian cycle and "
                 "sampled off-cycle moves eventually trap the snake (see the results table).", ""]
    training = f"- Phase `{phase}`; experiment file `experiments/{name}.env`"
    training += f"; write-up: {write_up_url}." if write_up_url else "."
    body += ["## Training", "", training, "- Resolved configuration (`config.json`):", "",
             "```json", json.dumps(run_cfg, indent=2), "```", ""]
    for figure in figure_names:
        body += [f"![{figure}](figures/{figure})", ""]
    libraries = ", ".join(f"{k} {v}" for k, v in versions.items() if k != "git_commit")
    body += ["## Provenance", "",
             f"- Code: {GITHUB} at commit `{versions.get('git_commit', 'unknown')}`.",
             f"- Library versions (`versions.json`): {libraries}.",
             "- `eval/summary.json` and `eval/eval_episodes.csv` are the files the repository's "
             "reports quote; every evaluated network is compared in "
             f"[reports/networks.md]({GITHUB}/blob/main/reports/networks.md).",
             *([f"- Collection: {collection_url}"] if collection_url else []), "",
             "## Files", "",
             f"- `{zip_name}`: the evaluated checkpoint in Stable-Baselines3's save format (policy "
             "weights and optimizer state, "
             "https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html).",
             "- `config.json`, `versions.json`: the run's resolved configuration and environment.",
             "- `eval/`: evaluation summary and one row per evaluation episode."]
    if phase == "train":
        body += ["- `train/progress.csv`: the SB3 training log; `figures/`: the learning curves."]
    body += ["", "## Licence", "", "MIT, like the repository.", ""]
    return "\n".join(_metadata(repo, board_tag, det, tags, base_model) + body)


def collection_note(run_cfg: dict, summary: dict, phase: str) -> str:
    """The note (500 characters at most) shown next to the repo inside the collection."""
    det = summary["modes"]["deterministic=True"]
    steps = det["steps_to_complete_mean"]
    steps_text = "" if steps is None else f", {steps:,.1f} steps"
    note = (f"{run_cfg['size']}^{run_cfg['ndim']} board: completion {det['success_rate_mean']:.3f}"
            f" +- {det['success_rate_std']:.3f} (argmax), fill {det['fill_mean']:.3f}{steps_text}; "
            f"{describe(phase, run_cfg)}.")
    return note[:NOTE_LIMIT]


# --- staging and uploading ---------------------------------------------------------------------
def _long_path(path: Path) -> str:
    """Absolute path with the ``\\\\?\\`` prefix on Windows: file calls work past MAX_PATH (260).

    https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
    """
    return f"\\\\?\\{path.resolve()}" if os.name == "nt" else str(path)


def stage(run_dir: Path, eval_dir: Path, dest: Path, figures_dir: Path) -> list[Path]:
    """Copy the checkpoint, configs, evaluation files and figures into ``dest`` (repo layout)."""
    name, zip_path = run_name(run_dir), model_file(run_dir)
    wanted = [(zip_path, dest / zip_path.name), (run_dir / "config.json", dest / "config.json"),
              (run_dir / "versions.json", dest / "versions.json"),
              (eval_dir / "summary.json", dest / "eval" / "summary.json"),
              (eval_dir / "eval_episodes.csv", dest / "eval" / "eval_episodes.csv"),
              (run_dir / "progress.csv", dest / "train" / "progress.csv")]
    wanted += [(figures_dir / f"{name}_{kind}.png", dest / "figures" / f"{name}_{kind}.png")
               for kind in ("curves", "fill_hist")]
    staged = []
    for source, target in wanted:
        if source.exists():  # progress.csv and figures exist for training runs only
            os.makedirs(_long_path(target.parent), exist_ok=True)
            shutil.copy2(_long_path(source), _long_path(target))
            staged.append(target)
    return staged


def hub_api():
    """The Hub client; ``huggingface_hub`` is imported lazily (optional ``hub`` group)."""
    from huggingface_hub import HfApi  # optional dependency group "hub"

    return HfApi()


def ensure_collection(api, cfg: Config):
    """The collection titled ``cfg.hf_collection`` in the namespace, created when it is missing."""
    for collection in api.list_collections(owner=cfg.hf_namespace):
        if collection.title == cfg.hf_collection:
            return collection
    return api.create_collection(title=cfg.hf_collection, namespace=cfg.hf_namespace,
                                 private=bool(cfg.hf_private),
                                 description=COLLECTION_DESCRIPTION[:DESCRIPTION_LIMIT - 1])


def publish_one(api, cfg: Config, name: str, staging: Path, collection_url: str,
                reports_dir: Path) -> dict:
    """Stage, card and upload one run; returns its ``published.json`` record."""
    runs_dir = Path(cfg.runs_dir)
    source, eval_dir = find_run(runs_dir, name), find_eval(runs_dir, name)
    if source is None or eval_dir is None:
        raise FileNotFoundError(f"run or evaluation of {name} not found under {runs_dir}")
    phase = source.name.split("_")[1]
    repo_id = f"{cfg.hf_namespace}/{repo_name(name)}"
    dest = staging / repo_name(name)
    files = stage(source, eval_dir, dest, reports_dir / "figures")
    run_cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    versions = json.loads((source / "versions.json").read_text(encoding="utf-8"))
    base = base_run(run_cfg)
    card = model_card(name, phase, run_cfg, summary, versions, repo_id, model_file(source).name,
                      base_model=f"{cfg.hf_namespace}/{repo_name(base)}" if base else None,
                      write_up_url=write_up(name, reports_dir), collection_url=collection_url,
                      figure_names=tuple(f.name for f in files if f.parent.name == "figures"))
    (dest / "README.md").write_text(card, encoding="utf-8", newline="\n")  # LF, not Windows CRLF
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=bool(cfg.hf_private))
    api.upload_folder(folder_path=str(dest), repo_id=repo_id, repo_type="model",
                      commit_message=f"{name}: checkpoint, config, evaluation files, model card")
    return {"name": name, "repo_id": repo_id, "url": f"{HUB}/{repo_id}",
            "note": collection_note(run_cfg, summary, phase),
            "files": [str(f.relative_to(dest)).replace("\\", "/") for f in files] + ["README.md"],
            "bytes": sum(os.stat(_long_path(f)).st_size for f in files)}


def run(cfg: Config, api=None, reports_dir: Path = REPORTS) -> Path:
    """Phase entry point: one Hub repo per run in ``cfg.publish_names``, all in one collection."""
    run_dir = make_run_dir(cfg, "publish")
    logger = setup_logging(run_dir)
    api = api or hub_api()
    collection = ensure_collection(api, cfg)
    logger.info("collection %s (%s)", collection.slug, collection.url)
    published = []
    for name in cfg.publish_names:
        try:
            record = publish_one(api, cfg, name, run_dir / "staging", collection.url, reports_dir)
        except Exception:  # noqa: BLE001 - one failed upload must not stop the others
            logger.exception("failed to publish %s", name)
            continue
        api.add_collection_item(collection.slug, item_id=record["repo_id"], item_type="model",
                                note=record["note"], exists_ok=True)
        logger.info("published %s -> %s (%d files, %s bytes)", name, record["url"],
                    len(record["files"]), f"{record['bytes']:,}")
        published.append(record)
    result = {"collection": {"title": cfg.hf_collection, "slug": collection.slug,
                             "url": collection.url}, "models": published}
    (run_dir / "published.json").write_text(json.dumps(result, indent=2), encoding="utf-8",
                                            newline="\n")
    logger.info("%d of %d runs published; %s", len(published), len(cfg.publish_names),
                collection.url)
    return run_dir
