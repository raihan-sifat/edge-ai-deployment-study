"""Verify the pushed repository state.

Run after pushing. Confirms the commit landed, every expected file is present,
and nothing that should have been ignored was uploaded.
"""

from __future__ import annotations

import json
import subprocess
import sys

REPO = "raihan-sifat/edge-ai-deployment-study"

KEY_FILES = {
    ".github/workflows/ci.yml": "CI workflow",
    ".pre-commit-config.yaml": "pre-commit hooks",
    ".gitignore": ".gitignore",
    "CITATION.cff": "citation metadata",
    "LICENSE": "license",
    "README.md": "README",
    "pyproject.toml": "packaging",
    "requirements.txt": "runtime deps",
    "configs/default.yaml": "default config",
    "configs/models.yaml": "model zoo",
    "configs/offline.yaml": "offline config",
    "docs/REPORT.md": "technical report",
    "docs/METHODOLOGY.md": "methodology",
    "docs/references.bib": "bibliography",
    "docs/index.html": "GitHub Pages page",
    "portfolio/case-study.mdx": "case study page",
    "portfolio/case-study.css": "case study styles",
    "portfolio/data/results.json": "portfolio metrics",
    "portfolio/images/accuracy_vs_latency.webp": "generated figure (webp)",
    "scripts/check_measurement_order.py": "measurement diagnostic",
    "scripts/export_portfolio_assets.py": "portfolio export",
    "scripts/validate_case_study.mjs": "MDX validator",
    "scripts/build_report.py": "report builder",
    "src/edgebench/pipeline.py": "pipeline",
    "src/edgebench/optim/quantization.py": "quantization",
    "src/edgebench/bench/latency.py": "latency protocol",
    "tests/test_optimizations.py": "optimization tests",
    "tests/test_benchmark.py": "benchmark tests",
}

UNWANTED_PREFIXES = (
    ".venv/",
    "node_modules/",
    "data/",
    "checkpoints/",
    "results/",
    "portfolio/node_modules/",
)


def run(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        print(f"command failed: {' '.join(command)}", file=sys.stderr)
        print(completed.stderr.strip(), file=sys.stderr)
    return completed.stdout


def main() -> int:
    print(f"Verifying {REPO}")
    print("=" * 62)

    # ---- 1. local and remote agree --------------------------------------
    local_head = run(["git", "rev-parse", "HEAD"]).strip()
    remote_refs = run(["git", "ls-remote", "origin"]).strip().splitlines()

    remote_heads = {}
    for line in remote_refs:
        sha, _, ref = line.partition("\t")
        remote_heads[ref.strip()] = sha.strip()

    remote_main = remote_heads.get("refs/heads/main")
    print()
    print("Commit")
    print(f"  local  HEAD          {local_head[:12]}")
    print(f"  remote refs/heads/main {remote_main[:12] if remote_main else '(none)'}")
    in_sync = bool(local_head) and local_head == remote_main
    print(f"  in sync              {'yes' if in_sync else 'NO'}")

    # ---- 2. file tree ----------------------------------------------------
    tree_json = run(["gh", "api", f"repos/{REPO}/git/trees/main?recursive=1"])
    try:
        tree = json.loads(tree_json)
    except json.JSONDecodeError:
        print()
        print("Could not read the remote file tree; is `gh` authenticated?", file=sys.stderr)
        return 1

    remote_files = {item["path"] for item in tree.get("tree", []) if item["type"] == "blob"}
    print()
    print(f"Files on remote: {len(remote_files)}")

    missing = [path for path in KEY_FILES if path not in remote_files]
    print()
    print("Key files")
    for path, label in KEY_FILES.items():
        mark = "OK " if path in remote_files else "MISSING"
        print(f"  {mark:8} {label:26} {path}")

    # ---- 3. nothing that should be ignored -------------------------------
    unwanted = sorted(path for path in remote_files if path.startswith(UNWANTED_PREFIXES))
    print()
    print("Ignored paths that leaked onto the remote")
    print(f"  {unwanted if unwanted else 'none'}")

    # ---- 4. CI status ----------------------------------------------------
    print()
    print("CI")
    runs = run(
        [
            "gh",
            "run",
            "list",
            "--repo",
            REPO,
            "--limit",
            "5",
            "--json",
            "status,conclusion,name,createdAt",
        ]
    )
    try:
        run_list = json.loads(runs) if runs.strip() else []
    except json.JSONDecodeError:
        run_list = []

    if not run_list:
        print("  no workflow runs yet (they can take a minute to appear)")
    else:
        for entry in run_list:
            conclusion = entry.get("conclusion") or entry.get("status")
            print(f"  {conclusion!s:12} {entry.get('name')}")

    # ---- 5. Pages --------------------------------------------------------
    print()
    print("GitHub Pages")
    pages = run(["gh", "api", f"repos/{REPO}/pages"])
    if '"html_url"' in pages:
        url = json.loads(pages).get("html_url")
        print(f"  enabled   {url}")
    else:
        print("  not enabled")
        print("  enable at: Settings -> Pages -> Source: 'Deploy from a branch',")
        print("             Branch: main, Folder: /docs")

    # ---- verdict ---------------------------------------------------------
    print()
    print("=" * 62)
    problems: list[str] = []
    if not in_sync:
        problems.append("local and remote are out of sync")
    if missing:
        problems.append(f"{len(missing)} key file(s) missing on the remote")
    if unwanted:
        problems.append(f"{len(unwanted)} ignored path(s) were pushed")

    if problems:
        print("ISSUES")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("All good: the push is complete and the remote matches the local history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
