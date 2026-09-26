#!/usr/bin/env python
"""Build the technical report as HTML and PDF from ``docs/REPORT.md``.

Pandoc is required. Rather than failing with a bare stack trace, this script
checks for each dependency and tells you exactly how to install what is missing,
because "the report did not build" is a frustrating thing to hit on a fresh clone.

Usage::

    python scripts/build_report.py                 # HTML + PDF
    python scripts/build_report.py --html-only     # HTML only
    python scripts/build_report.py --check         # report tooling status only

PDF generation additionally needs a LaTeX engine (or wkhtmltopdf / weasyprint).
PDF output is deliberately not committed to the repository -- the Markdown is the
source of truth, and a stale PDF is worse than no PDF.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
SOURCE = DOCS_DIR / "REPORT.md"
BIBLIOGRAPHY = DOCS_DIR / "references.bib"
HTML_OUTPUT = DOCS_DIR / "report.html"
PDF_OUTPUT = DOCS_DIR / "report.pdf"

#: LaTeX engines pandoc can drive, in preference order.
LATEX_ENGINES = ("tectonic", "xelatex", "lualatex", "pdflatex")
#: Non-LaTeX PDF routes, used when no LaTeX engine is installed.
HTML_PDF_ENGINES = ("weasyprint", "wkhtmltopdf")


def which(executable: str) -> str | None:
    return shutil.which(executable)


def check_tooling() -> tuple[bool, bool]:
    """Return ``(can_build_html, can_build_pdf)`` and print a status table.

    HTML is always buildable: pandoc is preferred, but the standard-library
    renderer in ``markdown_to_html.py`` needs nothing installed. Only PDF depends
    on external tooling.
    """
    pandoc = which("pandoc")
    latex = next((name for name in LATEX_ENGINES if which(name)), None)
    html_pdf = next((name for name in HTML_PDF_ENGINES if which(name)), None)

    print("Report tooling")
    print("-" * 52)
    print(f"  pandoc          {pandoc or 'not installed (using built-in renderer)'}")
    print(f"  LaTeX engine    {latex or 'none'}")
    print(f"  HTML->PDF       {html_pdf or 'none'}")
    print()

    if not (latex or html_pdf):
        print("HTML will be built. For a PDF, install a LaTeX engine or weasyprint:")
        print("  Windows   winget install --id MiKTeX.MiKTeX")
        print("  macOS     brew install --cask mactex-no-gui    # or: pip install weasyprint")
        print("  Linux     sudo apt install texlive-xetex       # or: pip install weasyprint")
        print()
        print("docs/REPORT.md is complete and self-contained meanwhile; GitHub renders")
        print("it directly, including its tables, and any Markdown editor can export it.")
        print()

    return True, bool(latex or html_pdf)


def run(command: list[str]) -> bool:
    """Run a command, streaming output. Returns True on success."""
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"$ {printable}")
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        print(f"  -> exited with code {completed.returncode}")
        return False
    return True


def build_html() -> bool:
    """Build the HTML report, preferring pandoc and falling back to pure Python."""
    if which("pandoc"):
        command = [
            "pandoc",
            str(SOURCE),
            "--from=markdown+tex_math_dollars+pipe_tables",
            "--to=html5",
            "--standalone",
            "--toc",
            "--toc-depth=2",
            "--number-sections",
            "--citeproc",
            f"--bibliography={BIBLIOGRAPHY}",
            "--metadata=title:Benchmarking Efficient Deep Learning Models for Edge AI Deployment",
            "--embed-resources",
            f"--output={HTML_OUTPUT}",
        ]
        if run(command):
            return True
        print("\npandoc failed; falling back to the standard-library renderer.\n")

    # Standard-library fallback: always available, no installation required.
    return run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "markdown_to_html.py"),
            str(SOURCE),
            str(HTML_OUTPUT),
        ]
    )


def build_pdf(latex_engine: str | None) -> bool:
    """Build the PDF, preferring a LaTeX engine and falling back to HTML renderers."""
    if latex_engine:
        command = [
            "pandoc",
            str(SOURCE),
            "--from=markdown+tex_math_dollars+pipe_tables",
            "--toc",
            "--toc-depth=2",
            "--number-sections",
            "--citeproc",
            f"--bibliography={BIBLIOGRAPHY}",
            "--pdf-engine",
            latex_engine,
            "--variable=geometry:margin=1in",
            "--variable=fontsize=11pt",
            "--variable=colorlinks=true",
            f"--output={PDF_OUTPUT}",
        ]
        if run(command):
            return True
        print("\nLaTeX route failed; trying an HTML-based PDF engine instead.\n")

    html_pdf_engine = next((name for name in HTML_PDF_ENGINES if which(name)), None)
    if html_pdf_engine:
        command = [
            "pandoc",
            str(SOURCE),
            "--from=markdown+tex_math_dollars+pipe_tables",
            "--standalone",
            "--toc",
            "--number-sections",
            "--citeproc",
            f"--bibliography={BIBLIOGRAPHY}",
            f"--pdf-engine={html_pdf_engine}",
            f"--output={PDF_OUTPUT}",
        ]
        if run(command):
            return True

    print("\nCould not produce a PDF. docs/REPORT.md is complete and self-contained;")
    print("GitHub renders it directly, and any Markdown editor can export it.")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-only", action="store_true", help="Skip PDF generation.")
    parser.add_argument("--pdf-only", action="store_true", help="Skip HTML generation.")
    parser.add_argument("--check", action="store_true", help="Only report tooling status.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if the PDF could not be built (for CI).",
    )
    args = parser.parse_args()

    if not SOURCE.exists():
        print(f"error: {SOURCE} not found", file=sys.stderr)
        return 1

    can_html, can_pdf = check_tooling()
    if args.check:
        return 0 if (can_html and can_pdf) else 1

    if not can_html:  # pragma: no cover - the built-in renderer is always available
        return 1

    latex_engine = next((name for name in LATEX_ENGINES if which(name)), None)

    html_ok = PDF_ok = True
    if not args.pdf_only:
        html_ok = build_html()
    if not args.html_only:
        PDF_ok = build_pdf(latex_engine)

    print()
    for path in (HTML_OUTPUT, PDF_OUTPUT):
        if path.exists():
            print(f"  wrote {path.relative_to(REPO_ROOT)}  ({path.stat().st_size / 1024:.0f} KiB)")

    # A missing PDF is a soft failure: it depends on external tooling that the
    # reader may not want to install. A missing HTML is a hard failure.
    if args.strict:
        return 0 if (html_ok and PDF_ok) else 1
    return 0 if html_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
