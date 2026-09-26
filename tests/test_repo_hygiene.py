"""Repository hygiene: config files must parse, and CI must be structurally sane.

These tests exist because of a real failure. A hand edit inserted 2-space-indented
YAML into ``.github/workflows/ci.yml`` while the rest of the file used 4-space
indentation. The result was a YAML parse error, which GitHub reports only as
"This run likely failed because of a workflow file issue" with **zero jobs** --
no line number, no stack trace. The workflow had already been pushed, so the
failure surfaced after the fact on GitHub rather than locally.

Two lessons are encoded here:

1. Mixed indentation in YAML is a syntax error, not a style problem, and it is
   worth a test.
2. Every ``uses:`` step is pinned to a major version tag so a breaking change
   upstream cannot silently alter what CI runs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
CONFIG_DIR = REPO_ROOT / "configs"

# Every YAML document that ships with the project.
YAML_FILES = sorted(
    [
        *WORKFLOW_DIR.glob("*.yml"),
        *WORKFLOW_DIR.glob("*.yaml"),
        *CONFIG_DIR.glob("*.yml"),
        *CONFIG_DIR.glob("*.yaml"),
        REPO_ROOT / ".pre-commit-config.yaml",
    ]
)


def test_yaml_files_were_found():
    """Guard against the globs silently matching nothing."""
    assert len(YAML_FILES) >= 6, f"expected several YAML files, found {len(YAML_FILES)}"


@pytest.mark.parametrize("path", YAML_FILES, ids=lambda p: p.name)
def test_yaml_parses(path: Path):
    """Every shipped YAML file must parse."""
    assert path.exists(), f"{path} is missing"
    try:
        yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        pytest.fail(f"{path.relative_to(REPO_ROOT)} is not valid YAML:\n{error}")


@pytest.mark.parametrize("path", YAML_FILES, ids=lambda p: p.name)
def test_no_tabs_in_yaml(path: Path):
    """Tabs are illegal as YAML indentation and some editors insert them."""
    text = path.read_text(encoding="utf-8")
    offending = [
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if line.startswith("\t") or re.match(r"^\s*\t", line)
    ]
    assert not offending, f"{path.name} has tab-indented line(s): {offending}"


@pytest.mark.parametrize("path", YAML_FILES, ids=lambda p: p.name)
def test_yaml_indentation_is_consistent(path: Path):
    """Sibling mapping keys must share a column, and sequences must align.

    This is a structural check, done on the composed node tree rather than on raw
    line widths, because width-only heuristics are wrong. A legitimate file can
    use width 4 for a block-sequence dash and width 6 for that item's own keys::

        optimizations:
            - id: fp32
              enabled: true

    Both 4 and 6 appear, yet the file is perfectly consistent. An earlier version
    of this test asserted every width was a multiple of the smallest and failed on
    every shipped config file for that reason.

    YAML already rejects *sibling keys at different columns* as a parse error, so
    a successful parse covers the case that broke CI. What this test adds is the
    sequence check, plus a clearer failure message than a bare parser error.
    """

    def walk(node, trail: str) -> None:
        if isinstance(node, yaml.MappingNode):
            if not node.flow_style:
                columns = {key.start_mark.column for key, _ in node.value}
                assert len(columns) == 1, (
                    f"{path.name}: mapping keys under {trail or '<root>'} sit at "
                    f"columns {sorted(columns)}; siblings must align"
                )
            for key, value in node.value:
                walk(value, f"{trail}.{key.value}" if trail else str(key.value))

        elif isinstance(node, yaml.SequenceNode):
            # Only block sequences have an alignment invariant. A flow sequence
            # like `[32, 224]` reports each item's column at its position on the
            # line, which is legitimate and must not be flagged.
            if not node.flow_style:
                columns = {item.start_mark.column for item in node.value}
                assert len(columns) == 1, (
                    f"{path.name}: sequence items under {trail or '<root>'} start at "
                    f"columns {sorted(columns)}; they must align"
                )
            for index, item in enumerate(node.value):
                walk(item, f"{trail}[{index}]")

    composed = yaml.compose(path.read_text(encoding="utf-8"))
    if composed is not None:
        walk(composed, "")


# ---------------------------------------------------------------------------
# Workflow structure
# ---------------------------------------------------------------------------

WORKFLOWS = sorted(WORKFLOW_DIR.glob("*.yml"))


def test_workflows_exist():
    assert WORKFLOWS, "no workflow files found"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_has_jobs(path: Path):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path.name} must define a mapping"
    jobs = document.get("jobs")
    assert isinstance(jobs, dict) and jobs, f"{path.name} defines no jobs"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_job_has_runner_and_steps(path: Path):
    """A job without these is the other way a workflow silently does nothing."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job_name, job in document["jobs"].items():
        assert isinstance(job, dict), f"{path.name}:{job_name} must be a mapping"
        assert "runs-on" in job, f"{path.name}:{job_name} has no runs-on"
        steps = job.get("steps")
        assert isinstance(steps, list) and steps, f"{path.name}:{job_name} has no steps"
        for index, step in enumerate(steps):
            assert isinstance(step, dict), f"{path.name}:{job_name} step {index} is not a mapping"
            assert "uses" in step or "run" in step, (
                f"{path.name}:{job_name} step {index} has neither 'uses' nor 'run'"
            )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_needs_references_exist(path: Path):
    """A typo'd `needs:` silently prevents a job from ever running."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    job_names = set(document["jobs"])
    for job_name, job in document["jobs"].items():
        needs = job.get("needs")
        if needs is None:
            continue
        targets = [needs] if isinstance(needs, str) else list(needs)
        for target in targets:
            assert target in job_names, (
                f"{path.name}:{job_name} needs {target!r}, which is not a job in this file"
            )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_actions_are_version_pinned(path: Path):
    """`uses:` must carry an explicit version, never a moving branch."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    unpinned: list[str] = []

    for job in document["jobs"].values():
        for step in job.get("steps", []):
            reference = step.get("uses")
            if not reference:
                continue
            if reference.startswith("./"):
                continue
            _, _, version = reference.partition("@")
            if not version or version in {"main", "master", "HEAD"}:
                unpinned.append(reference)

    assert not unpinned, f"{path.name} has unpinned action(s): {unpinned}"


def test_ci_covers_the_python_floor():
    """CI must exercise the lowest supported Python, per pyproject requires-python."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'requires-python\s*=\s*">=\s*([\d.]+)"', text)
    assert match, "could not find requires-python in pyproject.toml"
    floor = match.group(1)

    document = yaml.safe_load((WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8"))
    versions: list[str] = []
    for job in document["jobs"].values():
        matrix = (job.get("strategy") or {}).get("matrix") or {}
        versions.extend(str(v) for v in matrix.get("python-version", []))
        for step in job.get("steps", []):
            python_version = (step.get("with") or {}).get("python-version")
            if python_version:
                versions.append(str(python_version))

    minor = ".".join(floor.split(".")[:2])
    assert minor in versions, (
        f"pyproject requires >= {floor} but CI never tests {minor}; "
        f"tested versions were {sorted(set(versions))}"
    )
