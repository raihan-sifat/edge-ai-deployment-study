#!/usr/bin/env python
"""Render ``docs/REPORT.md`` to HTML using only the standard library.

Pandoc produces a nicer document, but requiring it means the report has no
artifact at all on a fresh machine. This fallback always works, so there is
always something to link to and print from a browser.

It is intentionally a *small* Markdown subset -- headings, paragraphs, the pipe
tables this report actually uses, fenced code, lists, blockquotes, inline code,
bold/italic, links, and display math delimited by ``$$``. It is not a general
Markdown implementation and does not try to be.

Usage::

    python scripts/markdown_to_html.py docs/REPORT.md docs/report.html
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

#: Compiled once rather than re-compiled per line. Markdown list markers.
BULLET_RE = re.compile(r"^[-*]\s+")
ORDERED_RE = re.compile(r"^\d+\.\s+")

CSS = """
:root {
  --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --border: #d1d9e0;
  --code-bg: #f6f8fa; --accent: #0969da; --table-stripe: #f6f8fa;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --border: #3d444d;
    --code-bg: #151b23; --accent: #4493f8; --table-stripe: #151b23;
  }
}
* { box-sizing: border-box; }
body {
  max-width: 46rem; margin: 0 auto; padding: 3rem 1.5rem 6rem;
  background: var(--bg); color: var(--fg);
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}
h1, h2, h3, h4 { line-height: 1.25; margin: 2.2rem 0 0.8rem; font-weight: 600; }
h1 { font-size: 2rem; border-bottom: 1px solid var(--border); padding-bottom: .4rem; }
h2 { font-size: 1.45rem; border-bottom: 1px solid var(--border); padding-bottom: .3rem; }
h3 { font-size: 1.15rem; }
h4 { font-size: 1rem; color: var(--muted); }
p, li { margin: 0.7rem 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code {
  background: var(--code-bg); padding: .18em .38em; border-radius: 5px;
  font: 0.875em/1.5 ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
pre {
  background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px;
  padding: 1rem; overflow-x: auto; margin: 1.1rem 0;
}
pre code { background: none; padding: 0; font-size: .85rem; }
table { border-collapse: collapse; width: 100%; margin: 1.2rem 0; font-size: .875rem; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--border); padding: .5rem .75rem; text-align: left; vertical-align: top; }
th { background: var(--code-bg); font-weight: 600; }
tbody tr:nth-child(even) { background: var(--table-stripe); }
blockquote {
  border-left: 3px solid var(--border); margin: 1.2rem 0; padding: .1rem 1rem;
  color: var(--muted);
}
hr { border: none; border-top: 1px solid var(--border); margin: 2.5rem 0; }
.math { display: block; text-align: center; margin: 1.4rem 0; font-style: italic; }
.title-block { margin-bottom: 2rem; }
.title-block .subtitle { color: var(--muted); font-size: 1.08rem; }
.title-block .author { color: var(--muted); font-size: .9rem; margin-top: .6rem; }
"""


def _inline(text: str) -> str:
    """Apply inline formatting to already-escaped text."""
    # Inline code first, so its content is not further transformed.
    code_spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        code_spans.append(match.group(1))
        return f"\x00{len(code_spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    def unstash(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return f"<code>{code_spans[index]}</code>"

    return re.sub(r"\x00(\d+)\x00", unstash, text)


def _table(rows: list[str]) -> str:
    """Render GitHub-style pipe-table rows to an HTML table."""
    parsed: list[list[str]] = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        parsed.append(cells)

    # Row index 1 is the alignment separator.
    header, body = parsed[0], parsed[2:]
    parts = ["<table>", "<thead><tr>"]
    parts += [f"<th>{_inline(html.escape(cell))}</th>" for cell in header]
    parts.append("</tr></thead>")
    if body:
        parts.append("<tbody>")
        for row in body:
            parts.append("<tr>")
            parts += [f"<td>{_inline(html.escape(cell))}</td>" for cell in row]
            parts.append("</tr>")
        parts.append("</tbody>")
    parts.append("</table>")
    return "\n".join(parts)


def convert(markdown: str) -> str:
    """Convert the Markdown subset used by this project's report."""
    lines = markdown.splitlines()
    output: list[str] = []

    in_code = False
    in_table = False
    in_list = False
    table_rows: list[str] = []
    paragraph: list[str] = []
    front_matter: dict[str, str] = {}

    def flush_paragraph() -> None:
        if paragraph:
            output.append(f"<p>{_inline(html.escape(' '.join(paragraph)))}</p>")
            paragraph.clear()

    def flush_table() -> None:
        nonlocal in_table
        if table_rows:
            output.append(_table(table_rows))
            table_rows.clear()
        in_table = False

    def flush_list() -> None:
        nonlocal in_list
        if in_list:
            output.append("</ul>")
            in_list = False

    index = 0
    # Strip YAML front matter, capturing it for the title block.
    if lines and lines[0].strip() == "---":
        index = 1
        while index < len(lines) and lines[index].strip() != "---":
            if ":" in lines[index]:
                key, _, value = lines[index].partition(":")
                front_matter[key.strip()] = value.strip().strip('"')
            index += 1
        index += 1

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            flush_table()
            flush_list()
            if in_code:
                output.append("</code></pre>")
                in_code = False
            else:
                output.append("<pre><code>")
                in_code = True
            index += 1
            continue

        if in_code:
            output.append(html.escape(line))
            index += 1
            continue

        if not stripped:
            flush_paragraph()
            flush_table()
            flush_list()
            index += 1
            continue

        if stripped == "---":
            flush_paragraph()
            flush_table()
            flush_list()
            output.append("<hr>")
            index += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            flush_list()
            in_table = True
            table_rows.append(stripped)
            index += 1
            continue

        flush_table()

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            flush_list()
            level = len(heading.group(1))
            output.append(f"<h{level}>{_inline(html.escape(heading.group(2)))}</h{level}>")
            index += 1
            continue

        if stripped.startswith("$$"):
            flush_paragraph()
            flush_list()
            math_lines: list[str] = []
            # Support both $$...$$ on one line and a multi-line block.
            remainder = stripped[2:]
            if remainder.endswith("$$") and len(remainder) > 2:
                math_lines.append(remainder[:-2].strip())
            else:
                if remainder:
                    math_lines.append(remainder)
                index += 1
                while index < len(lines) and "$$" not in lines[index]:
                    math_lines.append(lines[index])
                    index += 1
                if index < len(lines):
                    tail = lines[index].split("$$", 1)[0]
                    if tail.strip():
                        math_lines.append(tail)
            output.append(f'<div class="math">{html.escape(" ".join(math_lines).strip())}</div>')
            index += 1
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            flush_list()
            quote = re.sub(r"^>\s?", "", stripped)
            output.append(f"<blockquote><p>{_inline(html.escape(quote))}</p></blockquote>")
            index += 1
            continue

        if BULLET_RE.match(stripped):
            flush_paragraph()
            if not in_list:
                output.append("<ul>")
                in_list = True
            # The substitution is hoisted out of the f-string on purpose. A
            # backslash inside an f-string's expression part is a SyntaxError
            # before Python 3.12 (PEP 701), and this project supports 3.10.
            item = BULLET_RE.sub("", stripped)
            output.append(f"<li>{_inline(html.escape(item))}</li>")
            index += 1
            continue

        if ORDERED_RE.match(stripped):
            flush_paragraph()
            if not in_list:
                output.append('<ul style="list-style: decimal">')
                in_list = True
            item = ORDERED_RE.sub("", stripped)
            output.append(f"<li>{_inline(html.escape(item))}</li>")
            index += 1
            continue

        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    flush_table()
    flush_list()
    if in_code:
        output.append("</code></pre>")

    title = front_matter.get("title", "Technical Report")
    title_block = [
        '<div class="title-block">',
        f"<h1>{html.escape(title)}</h1>",
    ]
    if front_matter.get("subtitle"):
        title_block.append(f'<p class="subtitle">{html.escape(front_matter["subtitle"])}</p>')
    if front_matter.get("author"):
        title_block.append(
            f'<p class="author">{html.escape(front_matter["author"])} &middot; 2026</p>'
        )
    title_block.append("</div>")

    body = "\n".join(output)
    # The front-matter title is rendered separately; drop the duplicated H1.
    body = re.sub(r"^<h1>.*?</h1>\s*", "", body, count=1)

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n"
        + "\n".join(title_block)
        + "\n"
        + body
        + "\n</body>\n</html>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    if not args.source.exists():
        print(f"error: {args.source} not found")
        return 1

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(convert(args.source.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"wrote {args.destination} ({args.destination.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
