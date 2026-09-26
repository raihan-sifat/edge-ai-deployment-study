# Portfolio case study

A drop-in case study page for your portfolio site, plus the figures and metrics it
renders from. Built for **Astro or Next.js** and deliberately dependency-free — no
Tailwind, no component library, no build-time image pipeline.

```
portfolio/
├── case-study.mdx     the page
├── case-study.css     its styles (self-contained, palette-aware)
├── data/
│   └── results.json   generated metrics — the page renders from this
└── images/            generated figures, WebP + PNG
```

Everything under `data/` and `images/` is **generated**. Edit `case-study.mdx` and
`case-study.css` freely; both are yours.

---

## 1. Generate the assets

```bash
python scripts/export_portfolio_assets.py                  # reads results/
python scripts/export_portfolio_assets.py --results results/offline   # or any run
```

This copies the figures into `portfolio/images/` as WebP (roughly 60% smaller than
PNG — 95 KiB → 41 KiB on the largest figure) with PNG fallbacks, and writes
`portfolio/data/results.json` containing the headline table, the environment, and
which optimizations were unavailable.

Re-run it after every benchmark run. The case study never contains a hardcoded
number, so it cannot drift from the results.

### It refuses to let you publish placeholder data

If the results came from the synthetic offline config, `results.json` gets
`"provisional": true`, the export prints a warning, and **the case study renders a
visible banner**:

> **Provisional numbers.** These numbers were produced on synthetic random tensors to
> validate the harness end to end. They are not measurements of model quality and must
> not be cited.

Run the real study (`edgebench run-all`) and re-export to clear it. Delete the
banner block from the MDX if you want it gone permanently — it is clearly marked.

---

## 2. Install into your site

### Astro

```bash
mkdir -p src/pages/projects
cp portfolio/case-study.mdx src/pages/projects/edge-ai-benchmarking.mdx
cp -r portfolio/data portfolio/images src/pages/projects/
cp portfolio/case-study.css src/styles/
```

Then, in the MDX front matter, point `layout` at your own layout and drop the
duplicate imports that the copy introduced:

```yaml
layout: "../../layouts/CaseStudy.astro" # edit to match your tree
```

Import the stylesheet in the page (Astro supports CSS imports from MDX):

```mdx
import "../styles/case-study.css";
```

If you would rather not use a layout file, delete the `layout` line entirely —
Astro will render the MDX inside your default layout.

### Next.js (App Router)

```bash
mkdir -p app/projects/edge-ai-benchmarking
cp portfolio/case-study.mdx app/projects/edge-ai-benchmarking/page.mdx
cp -r portfolio/data portfolio/images app/projects/edge-ai-benchmarking/
cp portfolio/case-study.css app/projects/edge-ai-benchmarking/
```

Two changes to the front matter:

1. **Delete** the `layout:` line — Next does not use it.
2. **Uncomment** `slug:` if your MDX setup reads slugs from front matter.

Then add the CSS import at the top of the MDX, after the front matter:

```mdx
import "./case-study.css";
```

#### Next.js also needs frontmatter plugins — this is verified, not a guess

`@next/mdx` does **not** parse YAML front matter by default. Without a plugin the
build still _succeeds_, but the page renders its own front matter as visible body
text at the top:

> --- title: "Benchmarking Efficient Models for Edge AI" description: "I built a
> reproducible CPU harness…" … layout: "../../layouts/CaseStudy.astro" ---

The `---` lines parse as thematic breaks and the YAML keys become ordinary
paragraphs, which is valid Markdown — so nothing errors. It just looks broken.
`scripts/validate_case_study.mjs` detects exactly this and is the reason the
requirement is stated here.

Add the plugins to `next.config.mjs`:

```js
import createMDX from "@next/mdx";
import remarkFrontmatter from "remark-frontmatter";
import remarkMdxFrontmatter from "remark-mdx-frontmatter";

const withMDX = createMDX({
    options: {
        remarkPlugins: [
            remarkFrontmatter,
            [remarkMdxFrontmatter, { name: "frontmatter" }],
        ],
    },
});

export default withMDX({ pageExtensions: ["js", "jsx", "md", "mdx"] });
```

Then install them:

```bash
npm install remark-frontmatter remark-mdx-frontmatter
```

The JSON import works with no extra plugins.

### Anything else

The MDX is plain Markdown with JSX expressions and a JSON import, which is the
common denominator of essentially every MDX pipeline (Vite, Remix, Gatsby, Eleventy
with a plugin). The CSS is unlayered and scoped under `.cs-`, so it will not
collide with an existing stylesheet.

---

## 3. Match your site

### Palette

The CSS reads your site's tokens when they exist, and falls back otherwise:

```css
--cs-fg: var(--color-text, #1f2328);
--cs-fg-muted: var(--color-text-muted, #59636e);
--cs-bg: var(--color-bg, #ffffff);
--cs-bg-soft: var(--color-bg-subtle, #f6f8fa);
--cs-border: var(--color-border, #d1d9e0);
--cs-accent: var(--color-accent, #0969da);
```

So if your site already defines `--color-text` and friends, **it inherits your
theme automatically.** Rename the tokens at the top of `case-study.css` if yours
are named differently.

The success/failure colours (`--cs-good`, `--cs-bad`) and the warning banner are
hardcoded because they carry meaning. Change them if you like, but keep them
distinguishable — a red speedup is the point of the table.

### Dark mode

Handled via `prefers-color-scheme`. If your site uses a **class-based** toggle
(`<html class="dark">`) rather than the OS preference, add the class selector to
the dark-mode block:

```css
@media (prefers-color-scheme: dark) {
    /* ... */
}
html.dark {
    /* duplicate the token overrides here */
}
```

### Typography

The CSS sets no `font-family` on body text, on purpose — your site's typography
applies. Monospace is only used where it earns its place: numbers, code, and the
stat values.

---

## 4. Before you publish

Placeholders to replace in `case-study.mdx`:

| Placeholder               | Where                                                               |
| ------------------------- | ------------------------------------------------------------------- |
| `raihan-sifat`            | front matter `repo`, `projectPage`, `report`, and the Links section |
| The canonical project URL | Links section — points at your GitHub Pages page                    |
| `layout:`                 | front matter (Astro only)                                           |

Also worth a check:

- [ ] Re-run `export_portfolio_assets.py` against **real** CIFAR-10 results, and
      confirm the provisional banner is gone
- [ ] Confirm `results.environment.cpu` matches the machine you want to cite
- [ ] Click through the four links in the Links section
- [ ] View it at ~360px wide — the table scrolls horizontally by design, so make
      sure that reads acceptably
- [ ] Check dark mode if your site has it

---

## 5. Optional: React components

The MDX uses plain JSX expressions, so it needs no React integration and works in
Astro _without_ `@astrojs/react`. If you would rather have typed components, the
data is already in the right shape:

```tsx
// components/SpeedupBadge.tsx
type Props = { speedup: number | null };

export function SpeedupBadge({ speedup }: Props) {
    if (speedup === null) return <span className="cs-mute">—</span>;
    const tone =
        speedup >= 1.05 ? "cs-good" : speedup < 0.95 ? "cs-bad" : "cs-mute";
    return <span className={tone}>{speedup.toFixed(2)}×</span>;
}
```

Then replace the inline expression in the table with `<SpeedupBadge speedup={row.speedup} />`.
The `results.json` shape is documented below so you can type it precisely.

---

## `results.json` shape

```jsonc
{
    "provisional": true, // false once real data is exported
    "provisional_reason": "…", // null when not provisional
    "dataset": "cifar10", // "synthetic" triggers the banner
    "generated_at": "2026-09-26T…",

    "environment": {
        "cpu": "12th Gen Intel(R) Core(TM) i5-12450H",
        "platform": "Windows-11-…",
        "logical_cores": 12,
        "torch": "2.14.0+cpu",
        "python": "3.14.5",
        "onnxruntime": "1.30.0",
        "git_commit": "a1b2c3d",
    },

    "implementation": {
        "architectures": 5,
        "optimizations": 10,
        "resolutions": 2,
        "batch_sizes": 3,
        "thread_counts": 2,
        "latency_cells_per_config": 12,
        "tests": 222, // counted from pytest, not hardcoded
    },

    "headline_config": { "resolution": 32, "batch_size": 1, "num_threads": 1 },

    "models": [
        {
            "id": "resnet18",
            "name": "ResNet-18",
            "params_millions": 11.17,
            "macs_millions": 555.42,
            "family": "cnn",
            "adapt": "cifar",
        },
    ],

    "headline": [
        {
            "model_id": "resnet18",
            "model": "ResNet-18",
            "optimization_id": "static_int8",
            "optimization": "Static INT8 (PTQ)",
            "status": "applied", // applied | unavailable | failed
            "top1_pct": 10.74,
            "top1_delta_pp": -0.98,
            "weight_mib": 10.78,
            "size_reduction_pct": 74.75,
            "latency_ms": 5.75,
            "speedup": 3.12,
            "ece": 0.06,
            "timing": "stable", // or "unstable (cv=34%)"
            "unstable": false,
        },
    ],

    "best_per_model": [
        {
            "model": "…",
            "optimization": "…",
            "speedup": 3.12,
            "size_reduction_pct": 74.75,
            "top1_delta_pp": -0.98,
        },
    ],

    "unavailable": [
        {
            "model": "ResNet-18",
            "optimization": "torch.compile",
            "reason": "no working torch.compile backend…",
        },
    ],

    "figures": {
        "accuracy_vs_latency": {
            "available": true,
            "caption": "…",
            "alt": "…", // alt text ships with the figure
            "webp": "./images/accuracy_vs_latency.webp",
            "png": "./images/accuracy_vs_latency.png",
            "converted": true,
            "original_width": 1680,
        },
    },

    "counts": { "records": 8, "applied": 7, "not_applied": 1 },
}
```

Notes that matter when you consume it:

- **`null` is meaningful** and appears wherever a value could not be measured —
  unavailable optimizations have `null` accuracy and latency. Render it as `—`,
  not `0`.
- **`speedup` is relative to the same model's FP32 row**, not to a global baseline.
- **A figure with `available: false` was never generated.** The MDX guards on this
  so a partial run produces a shorter page rather than broken images.
- **`unstable: true`** means the cell's coefficient of variation exceeded 15%. The
  honest thing to do is show it.

---

## Regenerating after a new run

```bash
edgebench run-all                      # train, optimize, measure
python scripts/export_portfolio_assets.py   # refresh the page's data and images
```

No edits to `case-study.mdx` are needed — it renders whatever the export produced.
