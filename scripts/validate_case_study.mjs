// Validates portfolio/case-study.mdx against the real MDX compiler.
//
// Why this exists
// ---------------
// "It compiles" and "it renders correctly" are different things for MDX, and the
// gap is the frontmatter question. Without a frontmatter plugin the YAML block is
// *valid* MDX -- `---` parses as a thematic break and the key/value lines become
// paragraphs -- so compilation succeeds while the page visibly renders its own
// front matter as body text at the top. A naive "does it compile?" check gives a
// false pass, and the failure only shows up when you look at the rendered page.
//
// So this script checks three things per configuration:
//   1. does it compile at all
//   2. is the frontmatter exposed as data
//   3. did the YAML text leak into the compiled output
//
// Two configurations are tested, corresponding to the two host setups in
// portfolio/README.md:
//
//   - with remark-frontmatter + remark-mdx-frontmatter  -> Astro, Next.js (configured)
//   - with no remark plugins                            -> Next.js default config
//
// The second is *expected* to fail on the leak check. That is not a defect in the
// case study; it is the documented reason Next.js users must add the plugins.
//
// Setup (the packages are deliberately not project dependencies -- this is a
// portfolio asset check, not part of the benchmark harness):
//
//   npm install --no-save @mdx-js/mdx@^3 remark-frontmatter remark-mdx-frontmatter
//   node scripts/validate_case_study.mjs
//
// Exits 0 when the MDX is valid and frontmatter is consumed, 1 otherwise.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, "..");
const TARGET =
    process.argv[2] ?? path.join(REPO_ROOT, "portfolio", "case-study.mdx");

let compile;
let remarkFrontmatter;
let remarkMdxFrontmatter;

try {
    ({ compile } = await import("@mdx-js/mdx"));
    remarkFrontmatter = (await import("remark-frontmatter")).default;
    remarkMdxFrontmatter = (await import("remark-mdx-frontmatter")).default;
} catch (error) {
    console.error("Missing validation dependencies.\n");
    console.error(
        "  npm install --no-save @mdx-js/mdx@^3 remark-frontmatter remark-mdx-frontmatter",
    );
    console.error("  node scripts/validate_case_study.mjs\n");
    console.error(`Underlying error: ${error.message}`);
    process.exit(2);
}

const source = await readFile(TARGET, "utf8").catch((error) => {
    console.error(`Could not read ${TARGET}: ${error.message}`);
    process.exit(2);
});

// The JSON import needs a resolvable module to compile against, so it is stubbed.
//
// The stub must be `export const`, not `const`. MDX only treats lines beginning
// with `import` or `export` as ESM; a bare `const` is parsed as document content,
// where the following `{ ... }` becomes an MDX expression and acorn rejects it
// with a confusing "Could not parse expression" error.
const stubbed = source.replace(
    /^import\s+results\s+from\s+["'][^"']+["'];?$/m,
    "export const results = { provisional: false, headline: [], models: [], unavailable: [], counts: {}, implementation: {}, environment: {}, figures: {} };",
);

if (stubbed === source) {
    console.warn("warning: no `import results from ...` line found to stub");
}

// Distinctive strings from the front matter. If these survive into the compiled
// body, the front matter was rendered as content rather than consumed as data.
const FRONTMATTER_PROBES = [
    "layouts/CaseStudy.astro",
    "coverAlt",
    "projectPage",
    "featured",
];

async function attempt(label, remarkPlugins) {
    let code;
    try {
        const compiled = await compile(stubbed, { remarkPlugins, jsx: true });
        code = String(compiled.value);
    } catch (error) {
        const line = error.line ?? error.position?.start?.line;
        const column = error.column ?? error.position?.start?.column;
        console.log(`FAIL  ${label}`);
        console.log(`      compile error: ${error.message.split("\n")[0]}`);
        if (line) console.log(`      at line ${line}, column ${column ?? "?"}`);
        return { ok: false, exposesFrontmatter: false, leaked: null, bytes: 0 };
    }

    // Frontmatter becomes a data export, which lands near the top of the module.
    const exposesFrontmatter = /export\s+const\s+frontmatter\b/.test(
        code.slice(0, 4000),
    );

    // Restrict the leak search to the component body, so the frontmatter export
    // itself (which legitimately contains these strings) cannot trigger a match.
    const bodyStart = code.indexOf("_createMdxContent");
    const body = bodyStart === -1 ? code : code.slice(bodyStart);
    const leaked = FRONTMATTER_PROBES.filter((probe) => body.includes(probe));

    const ok = exposesFrontmatter && leaked.length === 0;
    console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
    console.log(
        `      compiles:            yes (${code.length.toLocaleString()} bytes)`,
    );
    console.log(
        `      frontmatter as data: ${exposesFrontmatter ? "yes" : "no"}`,
    );
    console.log(
        `      YAML leaked as text: ${leaked.length ? `yes -> ${leaked.join(", ")}` : "no"}`,
    );
    return {
        ok,
        exposesFrontmatter,
        leaked: leaked.length ? leaked : null,
        bytes: code.length,
    };
}

console.log(`Validating ${path.relative(REPO_ROOT, TARGET)}`);
console.log("-".repeat(68));

const configured = await attempt(
    "remark-frontmatter + remark-mdx-frontmatter",
    [remarkFrontmatter, [remarkMdxFrontmatter, { name: "frontmatter" }]],
);

const defaults = await attempt(
    "no remark plugins (Next.js default config)",
    [],
);

console.log("-".repeat(68));
console.log("VERDICT");
console.log();

if (configured.ok) {
    console.log(
        "  Astro:   works as-is (and Next.js, with the plugins below).",
    );
} else {
    console.log(
        "  Astro:   PROBLEM -- frontmatter is not consumed. See the error above.",
    );
}

console.log();

if (defaults.leaked) {
    console.log(
        "  Next.js: REQUIRES a frontmatter plugin, or the page renders its own",
    );
    console.log("  YAML as body text at the top. Add to next.config.mjs:");
    console.log();
    console.log("    const withMDX = createMDX({");
    console.log("      options: {");
    console.log("        remarkPlugins: [");
    console.log("          (await import('remark-frontmatter')).default,");
    console.log(
        "          [(await import('remark-mdx-frontmatter')).default, { name: 'frontmatter' }],",
    );
    console.log("        ],");
    console.log("      },");
    console.log("    });");
} else {
    console.log("  Next.js: works without extra plugins.");
}

console.log();

// The default-config leak is informational, not a failure: the MDX itself is fine.
process.exit(configured.ok ? 0 : 1);
