"""Guards on pyproject.toml that protect the licence and CUDA-wheel decisions permanently."""

import tomllib
from pathlib import Path

PYPROJECT = tomllib.loads(
    (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
)


def test_agpl_markdown_pdf_is_not_a_runtime_dependency():
    runtime = " ".join(PYPROJECT["project"]["dependencies"]).lower()
    assert "markdown-pdf" not in runtime and "pymupdf" not in runtime
    assert any("markdown-pdf" in d for d in PYPROJECT["dependency-groups"]["docs"])


def test_torch_comes_from_the_explicit_cu130_index():
    sources = PYPROJECT["tool"]["uv"]["sources"]["torch"]
    assert sources[0]["index"] == "pytorch-cu130"
    (index,) = [i for i in PYPROJECT["tool"]["uv"]["index"] if i["name"] == "pytorch-cu130"]
    assert index["url"] == "https://download.pytorch.org/whl/cu130" and index["explicit"] is True


def test_python_is_pinned_to_3_13():
    assert PYPROJECT["project"]["requires-python"] == ">=3.13,<3.14"


def test_hub_client_is_an_optional_group():
    runtime = " ".join(PYPROJECT["project"]["dependencies"]).lower()
    assert "huggingface" not in runtime
    assert any("huggingface-hub" in d for d in PYPROJECT["dependency-groups"]["hub"])
