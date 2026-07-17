"""Release metadata stays synchronized across the package and citation files."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

import raw2features

ROOT = Path(__file__).resolve().parents[1]


def test_release_versions_match() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    version = project["project"]["version"]

    assert raw2features.__version__ == version
    assert citation["version"] == version
    assert f"## [{version}]" in (ROOT / "CHANGELOG.md").read_text()
