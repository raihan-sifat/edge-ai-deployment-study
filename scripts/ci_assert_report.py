"""CI assertion: the HTML report was actually rendered.

``scripts/build_report.py`` exits zero when it cannot find pandoc, because the
built-in Python renderer is a valid fallback. That tolerance makes an exit-code
check meaningless here: the script would pass even if it wrote nothing at all.

So this asserts on the artifact: the file exists, is non-trivially sized, and
contains the structural elements the report depends on -- headings, at least one
table, and at least one code block. A renderer that silently dropped tables
would otherwise ship as "green".

Run from the repository root:

    python scripts/ci_assert_report.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPORT = Path("docs/report.html")

#: The rendered report is ~30 KiB. A file far below that means the renderer
#: produced a shell with no content in it.
MIN_BYTES = 8_000


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def main() -> int:
    if not REPORT.exists():
        fail(f"{REPORT} was not written")

    html = REPORT.read_text(encoding="utf-8")
    size = REPORT.stat().st_size
    print(f"{REPORT}: {size / 1024:.1f} KiB")

    if size < MIN_BYTES:
        fail(f"{REPORT} is only {size} bytes; expected at least {MIN_BYTES}")

    checks = [
        ("<html" in html.lower(), "an <html> element"),
        ("<body" in html.lower(), "a <body> element"),
        (bool(re.search(r"<h1[ >]", html)), "a level-1 heading"),
        (html.count("<h2") >= 8, f"at least 8 level-2 headings (found {html.count('<h2')})"),
        (html.count("<table") >= 1, f"at least 1 table (found {html.count('<table')})"),
        (
            html.count("<pre") >= 3,
            f"at least 3 code blocks (found {html.count('<pre')})",
        ),
        ("<li" in html, "at least one list item"),
    ]

    failed = [label for ok, label in checks if not ok]
    if failed:
        fail("the rendered report is missing: " + "; ".join(failed))

    # The report's own title must be present, proving the right document was read.
    if "Benchmarking Efficient Deep Learning Models" not in html:
        fail("the report title is missing; the wrong file may have been rendered")

    print(
        f"headings: {html.count('<h2')} h2, tables: {html.count('<table')}, "
        f"code blocks: {html.count('<pre')}"
    )
    print("report assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
