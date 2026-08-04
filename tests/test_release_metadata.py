"""Release metadata stays synchronized across the package and citation files."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml
from packaging.requirements import Requirement

import raw2features

ROOT = Path(__file__).resolve().parents[1]


def test_release_versions_match() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    version = project["project"]["version"]

    assert raw2features.__version__ == version
    assert citation["version"] == version
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert f"## [{version}]" in changelog
    if f"## [{version}] - Unreleased" in changelog:
        assert "date-released" not in citation


def test_kronos2_xformers_pin_only_targets_published_wheel_versions() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirement = Requirement(
        next(
            item
            for item in project["project"]["optional-dependencies"]["kronos2"]
            if item.startswith("xformers")
        )
    )
    assert str(requirement.specifier) == "==0.0.29.post3"
    assert requirement.marker is not None

    def selected(python_version: str) -> bool:
        return requirement.marker.evaluate(
            {
                "sys_platform": "linux",
                "platform_machine": "x86_64",
                "platform_python_implementation": "CPython",
                "python_version": python_version,
                "python_full_version": f"{python_version}.0",
            }
        )

    assert selected("3.11")
    assert selected("3.12")
    assert not selected("3.13")

    pypy_environment = {
        "sys_platform": "linux",
        "platform_machine": "x86_64",
        "platform_python_implementation": "PyPy",
        "python_version": "3.12",
        "python_full_version": "3.12.0",
    }
    assert not requirement.marker.evaluate(pypy_environment)
