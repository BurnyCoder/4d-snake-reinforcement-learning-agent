"""Tests for snake4d.publish on synthetic runs with a fake Hub client (no network, no token)."""

import json
from pathlib import Path
from types import SimpleNamespace

from snake4d import publish
from snake4d.config import Config

ROOT = Path(__file__).resolve().parents[1]
EXP05 = "tester/4d-snake-exp05-bc-4x4"
EXP05B = "tester/4d-snake-exp05b-ppo-4x4-from-bc"


def _summary(success=0.963, std=0.009, fill=0.994, steps=43.5, sampling=True):
    mode = {"success_rate_mean": success, "success_rate_std": std, "episodes": 300,
            "success_rate": success, "fill_mean": fill, "return_mean": 24.46, "length_mean": 43.1,
            "steps_to_complete_mean": steps,
            "win_within": {"C": 0.0, "2C": 0.01, "4C": success, "C2/2": success}}
    modes = {"deterministic=True": mode}
    if sampling:
        modes["deterministic=False"] = dict(mode)
    return {"board": "2^4", "n_cells": 16, "seeds": [0, 1, 2], "per_run": {}, "modes": modes}


def _runs(root: Path, name: str, phase: str = "train", zip_name: str = "best_model.zip",
          **cfg_kw) -> Path:
    """A train/imitate run dir plus its evaluation dir, in the layout the real phases write."""
    run_dir = root / f"20260904-120000_{phase}_{name}"
    run_dir.mkdir(parents=True)
    Config(size=2, ndim=4, **cfg_kw).to_json(run_dir / "config.json")
    versions = {"git_commit": "abc123", "torch": "2.14.0"}
    (run_dir / "versions.json").write_text(json.dumps(versions), encoding="utf-8")
    (run_dir / zip_name).write_bytes(b"not really a zip")
    if phase == "train":
        (run_dir / "progress.csv").write_text("time/total_timesteps\n1\n", encoding="utf-8")
    suffix = "eval" if phase == "imitate" else "best"
    eval_dir = root / f"20260904-130000_evaluate_{name}_{suffix}"
    eval_dir.mkdir()
    crlf = json.dumps(_summary(), indent=2).replace("\n", "\r\n")  # as Windows text mode writes it
    (eval_dir / "summary.json").write_bytes(crlf.encode("utf-8"))
    (eval_dir / "eval_episodes.csv").write_text(
        "seed,deterministic,success,fill,length,return\n0,1,1,1.0,40,24.0\n", encoding="utf-8")
    return run_dir


class FakeApi:
    """Records the HfApi calls the phase makes; returns Collection-like objects."""

    def __init__(self):
        self.calls, self.collections = [], []

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))

    def upload_folder(self, **kwargs):
        folder = Path(kwargs["folder_path"])
        files = sorted(str(p.relative_to(folder)).replace("\\", "/")
                       for p in folder.rglob("*") if p.is_file())
        self.calls.append(("upload_folder", kwargs["repo_id"], files))

    def list_collections(self, owner):
        return list(self.collections)

    def create_collection(self, title, namespace, private, description):
        slug = f"{namespace}/{title.lower().replace(' ', '-')}-0123"
        url = f"https://huggingface.co/collections/{slug}"
        collection = SimpleNamespace(slug=slug, title=title, url=url)
        self.collections.append(collection)
        self.calls.append(("create_collection", title, private))
        return collection

    def add_collection_item(self, collection_slug, item_id, item_type, note, exists_ok):
        self.calls.append(("add_collection_item", item_id, item_type, note, exists_ok))


def test_repo_name_write_up_base_run_and_model_file_fallback(tmp_path):
    name = "exp02d_ppo_2x4_backplay_strict"
    assert publish.repo_name(name) == "4d-snake-exp02d-ppo-2x4-backplay-strict"
    reports = ROOT / "reports"
    assert publish.write_up(name, reports).endswith("/exp02_ppo_2x4.md")
    assert publish.write_up("exp05b_ppo_4x4_from_bc", reports).endswith("/exp05_bc_4x4.md")
    assert publish.write_up("nothing", reports) is None
    clone_path = r"runs\20260904-221730_imitate_exp05_bc_4x4\bc_model.zip"
    assert publish.base_run({"model_path": clone_path}) == "exp05_bc_4x4"
    assert publish.base_run({"model_path": "exp05_bc_4x4"}) == "exp05_bc_4x4"  # run-name form
    assert publish.base_run({"model_path": ""}) is None
    for zip_name in ("final_model.zip", "bc_model.zip", "best_model.zip"):  # later ones win
        (tmp_path / zip_name).write_bytes(b"z")
        assert publish.model_file(tmp_path).name == zip_name


def test_model_card_metadata_body_and_negative_result():
    cfg = json.loads(json.dumps(Config(size=2, ndim=4).__dict__))
    versions = {"git_commit": "abc123", "torch": "2.14.0"}
    repo = "tester/4d-snake-exp02d-ppo-2x4-backplay-strict"
    card = publish.model_card("exp02d_ppo_2x4_backplay_strict", "train", cfg, _summary(), versions,
                              repo, "best_model.zip", write_up_url="https://example.org/exp02",
                              collection_url="https://example.org/c",
                              figure_names=("a_curves.png",))
    assert card.startswith("---\nlicense: mit\nlibrary_name: stable-baselines3\n")
    assert "value: 0.963 +/- 0.009" in card and "- backplay-curriculum" in card
    assert "base_model" not in card and "Negative result" not in card
    assert "| deterministic (argmax) | 0.963 +- 0.009 | 0.994 | 43.5 | 0.963 |" in card
    assert f"hf download {repo} best_model.zip" in card
    assert "![a_curves.png](figures/a_curves.png)" in card
    assert "https://example.org/exp02" in card and "https://example.org/c" in card
    assert all(ord(ch) < 128 for ch in card)
    failed = _summary(success=0.0, std=0.0, fill=0.401, steps=None, sampling=False)
    negative = publish.model_card("exp04_ppo_4x4", "train", cfg, failed, versions,
                                  "tester/4d-snake-exp04-ppo-4x4", "best_model.zip")
    assert "Negative result" in negative and "steps_to_complete" not in negative
    assert "| never |" in negative and "| sampling |" not in negative
    clone = publish.model_card("exp05b_ppo_4x4_from_bc", "train", cfg, _summary(), versions,
                               EXP05B, "best_model.zip", base_model=EXP05,
                               write_up_url="https://example.org/exp05")
    assert f"base_model: {EXP05}" in clone and "Use deterministic mode" in clone
    assert "analysis: https://example.org/exp05" in clone
    assert "the learning curves and the fill histogram" in clone
    assert "An MLP policy with two hidden layers of 512 units" in clone


def test_publish_phase_stages_uploads_and_collects(tmp_path):
    runs = tmp_path / "runs"
    clone = _runs(runs, "exp05_bc_4x4", phase="imitate", zip_name="bc_model.zip")
    _runs(runs, "exp05b_ppo_4x4_from_bc", model_path=str(clone / "bc_model.zip"))
    figures = tmp_path / "reports" / "figures"
    figures.mkdir(parents=True)
    (figures / "exp05b_ppo_4x4_from_bc_curves.png").write_bytes(b"png")
    cfg = Config(runs_dir=str(runs), hf_namespace="tester", hf_collection="Test nets",
                 publish_runs="exp05_bc_4x4,exp05b_ppo_4x4_from_bc,missing_run")
    api = FakeApi()
    run_dir = publish.run(cfg, api=api, reports_dir=tmp_path / "reports")

    created = [c[1] for c in api.calls if c[0] == "create_repo"]
    assert created == [EXP05, EXP05B]
    uploads = {c[1]: c[2] for c in api.calls if c[0] == "upload_folder"}
    assert uploads[EXP05] == ["README.md", "bc_model.zip", "config.json", "eval/eval_episodes.csv",
                              "eval/summary.json", "versions.json"]
    staged = set(uploads[EXP05B])
    assert {"figures/exp05b_ppo_4x4_from_bc_curves.png", "train/progress.csv"} <= staged
    staged_dir = run_dir / "staging" / EXP05B.split("/")[1]
    card_bytes = (staged_dir / "README.md").read_bytes()
    assert b"\r" not in card_bytes  # LF endings even on Windows
    card = card_bytes.decode("utf-8")
    assert f"base_model: {EXP05}" in card and "figures/exp05b_ppo_4x4_from_bc_curves.png" in card
    summary_bytes = (staged_dir / "eval" / "summary.json").read_bytes()
    assert b"\r" not in summary_bytes and json.loads(summary_bytes)["n_cells"] == 16  # CRLF -> LF
    clone_dir = run_dir / "staging" / EXP05.split("/")[1]
    clone_card = (clone_dir / "README.md").read_text(encoding="utf-8")
    assert "262,144 expert samples, 20 epochs, Adam 0.001" in clone_card  # n_envs * n_steps, config
    items = [c for c in api.calls if c[0] == "add_collection_item"]
    assert [i[1] for i in items] == created
    assert all(i[4] is True and len(i[3]) <= 500 for i in items)
    assert sum(c[0] == "create_collection" for c in api.calls) == 1
    result = json.loads((run_dir / "published.json").read_text(encoding="utf-8"))
    assert [m["repo_id"] for m in result["models"]] == created
    assert result["collection"]["slug"].startswith("tester/")
    assert "failed to publish missing_run" in (run_dir / "run.log").read_text(encoding="utf-8")

    publish.run(cfg, api=api, reports_dir=tmp_path / "reports")  # a re-run reuses the collection
    assert sum(c[0] == "create_collection" for c in api.calls) == 1


def test_publish_names_default_is_exactly_the_evaluated_networks():
    data = ROOT / "reports" / "data"
    evaluated = [p for p in data.iterdir() if p.name.endswith(("_best", "_eval"))]
    archived = {p.name.removesuffix("_best").removesuffix("_eval") for p in evaluated}
    assert archived == set(Config().publish_names) and len(archived) == 10


def test_collection_description_respects_the_hub_limit():
    assert len(publish.COLLECTION_DESCRIPTION) < publish.DESCRIPTION_LIMIT == 150
