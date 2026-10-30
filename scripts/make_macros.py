"""Generate paper/generated/*.tex and the README results block from results/*.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "generated"
BUDGETS = (128, 256, 512, 1024, 2048, 4096)
FRESH = {"beam": "BEAM-100K and 500K", "repomem-fresh": "RepoMem, keys", "repomem-fresh-plain": "RepoMem, no keys"}
STATIC = ("beam-static", "repomem-fresh-static")
METHODS = {
    "kanerva-ce": "\\kanerva + reranker", "kanerva": "\\kanerva", "kanerva-lex": "\\kanerva, lexical", "kanerva-r1": "First version",
    "rerank": "Hybrid + reranker", "hybrid": "Hybrid RRF", "hybrid-live": "Hybrid, live only", "hybrid-dedup": "Hybrid, newest of twins",
    "dense": "Dense", "bm25": "BM25", "mmr": "MMR", "gen-agents": "Generative Agents", "facility": "Facility location", "recency": "Recency",
}
PAIRS = {
    "kanerva vs hybrid": "\\kanerva vs hybrid", "kanerva vs rerank": "\\kanerva vs reranker", "kanerva vs kanerva-r1": "\\kanerva vs first version",
    "kanerva-ce vs rerank": "+ reranker vs reranker", "kanerva-lex vs bm25": "Lexical vs BM25",
}
ABLATIONS = {
    "kanerva vs -neighbours": "Wide pool and neighbours", "kanerva vs -density": "Cost density",
    "kanerva vs -validity": "Validity", "kanerva vs -supersession": "Near-twin rules",
}
DEV = {"lme": "LongMemEval-S", "locomo": "LoCoMo", "repomem": "RepoMem", "beam-dev": "BEAM-1M"}
DEV_ROWS = {"first-version": "First version", "hybrid": "Hybrid RRF", "rerank": "Hybrid + reranker", "bm25": "BM25",
            "kanerva/density": "\\kanerva", "kanerva-ce/density": "\\kanerva + reranker", "kanerva-lex/density": "\\kanerva, lexical"}


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def define(lines: list[str], key: str, value) -> None:
    lines.append(f"\\expandafter\\def\\csname {key}\\endcsname{{{value}}}")


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def signed(x: float) -> str:
    return f"{x:+.3f}"


def sci(x: float) -> str:
    exponent = len(str(int(x))) - 1
    return f"${x / 10 ** exponent:.1f}\\times10^{{{exponent}}}$"


def pvalue(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def mean_over_budgets(summary: dict, method: str, metric: str = "recall") -> float:
    return sum(summary[f"{method}@{b}"][metric] for b in BUDGETS) / len(BUDGETS)


def recall_table(data: dict) -> str:
    rows = ["\\begin{tabular}{@{}l" + "r" * (len(BUDGETS) + 1) + "@{}}\\toprule", "Method & " + " & ".join(map(str, BUDGETS)) + " & mean \\\\\\midrule"]
    for name, title in FRESH.items():
        summary = data[name]["summary"]
        rows.append(f"\\multicolumn{{{len(BUDGETS) + 2}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        methods = [m for m in METHODS if f"{m}@128" in summary]
        best = {b: max(summary[f"{m}@{b}"]["recall"] for m in methods) for b in BUDGETS}
        best_mean = max(mean_over_budgets(summary, m) for m in methods)
        for m in methods:
            cells = []
            for b in BUDGETS:
                value = summary[f"{m}@{b}"]["recall"]
                cells.append(f"\\textbf{{{fmt(value)}}}" if abs(value - best[b]) < 5e-4 else fmt(value))
            mean = mean_over_budgets(summary, m)
            cells.append(f"\\textbf{{{fmt(mean)}}}" if abs(mean - best_mean) < 5e-4 else fmt(mean))
            rows.append(f"{METHODS[m]} & " + " & ".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    rows[-1] = "\\bottomrule\\end{tabular}"
    return "\n".join(rows)


def tests_table(tests: dict, pairs: dict, budgets: tuple[str, ...]) -> str:
    rows = ["\\begin{tabular}{@{}lllrrrr@{}}\\toprule", "Dataset & Comparison & $B$ & $\\Delta$ recall & Unit CI & Units $+$ & $p$ \\\\\\midrule"]
    for name, title in FRESH.items():
        for pair, label in pairs.items():
            if pair not in tests[name]:
                continue
            for budget in budgets:
                t = tests[name][pair][budget]
                low, high = t["cluster_ci"]
                p = t.get("p_holm", t["p"])
                rows.append(f"{title} & {label} & {budget} & {signed(t['diff'])} & [{signed(low)}, {signed(high)}] & {t['units_positive']} & {pvalue(p)} \\\\")
        rows.append("\\addlinespace")
    rows[-1] = "\\bottomrule\\end{tabular}"
    return "\n".join(rows)


def stale_table(tests: dict, updates: dict | None) -> str:
    methods = ("kanerva", "kanerva-lex", "kanerva-r1", "hybrid-live", "hybrid-dedup", "hybrid", "rerank", "bm25")
    rows = ["\\begin{tabular}{@{}lrrrr@{}}\\toprule", "Method & \\multicolumn{2}{c}{Keys} & \\multicolumn{2}{c}{No keys} \\\\", " & inject & error & inject & error \\\\\\midrule"]
    for m in methods:
        cells = []
        for name in ("repomem-fresh", "repomem-fresh-plain"):
            row = tests[name]["stale_on_changed_facts"][m]
            cells += [fmt(row["stale_any"]), fmt(row["stale_error"])]
        rows.append(f"{METHODS[m]} & " + " & ".join(cells) + " \\\\")
    rows.append("\\bottomrule\\end{tabular}")
    return "\n".join(rows)


def dev_table(main: dict, static: dict | None) -> str:
    rows = ["\\begin{tabular}{@{}l" + "r" * (len(BUDGETS) + 1) + "@{}}\\toprule", "Method & " + " & ".join(map(str, BUDGETS)) + " & mean \\\\\\midrule"]
    for name, title in DEV.items():
        rows.append(f"\\multicolumn{{{len(BUDGETS) + 2}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for key, label in DEV_ROWS.items():
            per, mean = main[name][key]
            rows.append(f"{label} & " + " & ".join(fmt(v) for v in per) + f" & {fmt(mean)} \\\\")
        if static:
            per, mean = static[f"{name}-m2v"]["kanerva/density"]
            rows.append("\\kanerva, static embedder & " + " & ".join(fmt(v) for v in per) + f" & {fmt(mean)} \\\\")
        rows.append("\\addlinespace")
    rows[-1] = "\\bottomrule\\end{tabular}"
    return "\n".join(rows)


def ablation_table(data: dict) -> str:
    labels = {"-item": "Item", "-role": "Role and speaker", "-query-item": "Query--item overlap", "-neighbourhood": "Neighbourhood", "-time": "Recency and activation"}
    rows = ["\\begin{tabular}{@{}l" + "r" * len(DEV) + "@{}}\\toprule", "Group removed & " + " & ".join(DEV.values()) + " \\\\\\midrule"]
    for key, label in labels.items():
        rows.append(f"{label} & " + " & ".join(signed(data["all"][n][1] - data[key][n][1]) for n in DEV) + " \\\\")
    rows.append("\\bottomrule\\end{tabular}")
    return "\n".join(rows)


def update_readme(data: dict, tests: dict, static: dict, latency: dict) -> None:
    rows = ["| Dataset | Budget | Kanerva | Kanerva + reranker | Hybrid + reranker | Hybrid | BM25 |", "|---|---:|---:|---:|---:|---:|---:|"]
    titles = {"beam": "Long conversations", "repomem-fresh": "Code history, with keys", "repomem-fresh-plain": "Code history, no keys"}
    for name, title in titles.items():
        summary = data[name]["summary"]
        for b in (256, 1024, 4096):
            values = {m: summary[f"{m}@{b}"]["recall"] for m in ("kanerva", "kanerva-ce", "rerank", "hybrid", "bm25")}
            best = max(values.values())
            cells = [f"**{fmt(v)}**" if abs(v - best) < 5e-4 else fmt(v) for v in values.values()]
            rows.append(f"| {title} | {b} | " + " | ".join(cells) + " |")
    beam, plain = data["beam"]["summary"], data["repomem-fresh-plain"]["summary"]
    tests["repomem-fresh"]["stale_on_changed_facts"]
    unkeyed = tests["repomem-fresh-plain"]["stale_on_changed_facts"]
    lift = tests["beam"]["kanerva vs rerank"]["mean"]
    text = f"""<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/recall-dark.png">
  <img src="docs/assets/recall-light.png" alt="Evidence recovered at each token budget on held-out data" width="100%">
</picture>

Share of the needed evidence that made it into the context. The test data was held out: BEAM conversations and five repositories that no part of the method was fitted on, {data['beam']['meta']['queries']:,} and {data['repomem-fresh']['meta']['queries']:,} questions. These are retrieval metrics, measured with no LLM in the loop.

{chr(10).join(rows)}

- **Long conversations are where it shines.** Across budgets, Kanerva recovers {100 * lift['diff']:.0f} points more evidence than hybrid search with a cross-encoder reranker, and it needs no reranker to do it. At 1024 tokens it finds {fmt(beam['kanerva@1024']['recall'])} of the evidence against {fmt(beam['rerank@1024']['recall'])}.
- **Stale facts mostly stop.** On facts that changed, plain hybrid search injects an outdated version {100 * unkeyed['hybrid']['stale_any']:.1f}% of the time. Kanerva does so {100 * unkeyed['kanerva']['stale_any']:.1f}% of the time from text alone, and never when notes carry keys.
- **Where it loses.** On code history at large budgets, the same rule that removes stale notes also drops some useful look-alikes. At 4096 tokens without keys, Kanerva has {fmt(plain['kanerva@4096']['recall'])} against {fmt(plain['hybrid@4096']['recall'])} for hybrid. With five repositories the intervals are wide, so treat code-history differences of a few points as ties.

The numbers above use a small sentence-embedding model. The plugins run without one by default: lexical search alone reaches {fmt(beam['kanerva-lex@1024']['recall'])} on long conversations at 1024 tokens, and `KANERVA_EMBEDDER=static` reaches {fmt(static['beam-static']['summary']['kanerva@1024']['recall'])}. A query over 10,000 notes takes about {latency['lexical@10000']['p50_ms']:.0f} ms on an M1 Pro without embeddings and {latency['hybrid@10000']['p50_ms']:.0f} ms with them."""
    path = ROOT / "README.md"
    readme = path.read_text()
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    head, rest = readme.split(start)
    _, tail = rest.split(end)
    path.write_text(f"{head}{start}\n{text}\n{end}{tail}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by scripts/make_macros.py; do not edit.",
        "\\newcommand{\\R}[3]{\\csname r-#1-#2-#3\\endcsname}",
        "\\newcommand{\\V}[1]{\\csname v-#1\\endcsname}",
    ]
    data = {n: load(f"fresh_{n}") for n in (*FRESH, *STATIC)}
    for name, entry in data.items():
        define(lines, f"v-n-{name}", f"{entry['meta']['queries']:,}".replace(",", "{,}"))
        for key, row in entry["summary"].items():
            m, b = key.split("@")
            define(lines, f"r-{name}-{m}-{b}", fmt(row["recall"]))
            if row.get("answer") == row.get("answer"):
                define(lines, f"v-ans-{name}-{m}-{b}", fmt(row["answer"]))
            define(lines, f"v-lat-{name}-{m}-{b}", f"{row['latency_ms']:.0f}")
        for m in sorted({k.split("@")[0] for k in entry["summary"]}):
            if all(f"{m}@{b}" in entry["summary"] for b in BUDGETS):
                define(lines, f"r-{name}-{m}-mean", fmt(mean_over_budgets(entry["summary"], m)))

    tests = load("fresh_tests")
    for name in FRESH:
        for pair, entry in tests[name].items():
            if pair == "stale_on_changed_facts":
                for m, row in entry.items():
                    for k, v in row.items():
                        define(lines, f"v-st-{name}-{m}-{k}", fmt(v))
                continue
            key = pair.replace(" ", "_")
            for budget, t in entry.items():
                define(lines, f"v-d-{name}-{key}-{budget}", signed(t["diff"]))
                define(lines, f"v-ci-{name}-{key}-{budget}", f"[{signed(t['cluster_ci'][0])}, {signed(t['cluster_ci'][1])}]")
                define(lines, f"v-pos-{name}-{key}-{budget}", t["units_positive"])
    for name, entry in load("fresh_tracks")["results"].items():
        for key, value in entry.items():
            track, rest = key.split("/")
            m, b = rest.split("@")
            define(lines, f"v-track-{name}-{track}-{m}-{b}", fmt(value))
    (OUT / "table_fresh.tex").write_text(recall_table(data))
    (OUT / "table_fresh_tests.tex").write_text(tests_table(tests, PAIRS, ("128", "1024", "4096", "mean")))
    (OUT / "table_fresh_ablation.tex").write_text(tests_table(tests, ABLATIONS, ("1024", "mean")))
    (OUT / "table_stale.tex").write_text(stale_table(tests, load("dev_updates")))

    scale = load("regret_scale")
    for d, horizon in scale["horizon"].items():
        define(lines, f"v-horizon-{d}", sci(horizon))

    main_dev, static_dev = load("dev_models"), load("dev_models_static")
    if main_dev:
        (OUT / "table_dev.tex").write_text(dev_table(main_dev["results"], static_dev["results"] if static_dev else None))
        for name, rows in main_dev["results"].items():
            for key, (per, mean) in rows.items():
                short = key.split("/")[0]
                define(lines, f"v-dev-{name}-{short}-mean", fmt(mean))
                for b, v in zip(BUDGETS, per):
                    define(lines, f"v-dev-{name}-{short}-{b}", fmt(v))
    rules = load("dev_models_rules")
    if rules:
        for name, rows in rules["results"].items():
            for key, (per, mean) in rows.items():
                if key.startswith("kanerva/"):
                    define(lines, f"v-rule-{name}-{key.split('/')[1]}", fmt(mean))
    ablation = load("dev_ablation")
    (OUT / "table_dev_ablation.tex").write_text(ablation_table(ablation["results"]))
    pool = load("dev_pool")["results"]
    for name, row in pool.items():
        for key in ("pool100", "pool200", "nb50x1", "prf", "size100", "size200", "nb50x1_size"):
            define(lines, f"v-pool-{name}-{key}", fmt(row[key]) if row[key] <= 1 else f"{row[key]:.0f}")

    updates = load("dev_updates")
    if updates:
        for key, row in updates["summary"]["lme-updates"].items():
            m, b = key.split("@")
            if b == "1024":
                define(lines, f"v-up-{m}-recall", fmt(row["recall"]))
                define(lines, f"v-up-{m}-inject", fmt(row["stale_any"]))
                define(lines, f"v-up-{m}-error", fmt(row["stale_error"]))

    systems = load("systems_mem0")
    define(lines, "v-mem0n", systems["meta"]["questions"])
    define(lines, "v-mem0add", f"{systems['meta']['add_ms_per_turn']:.0f}")
    for b, row in systems["same_questions"].items():
        for k, v in row.items():
            define(lines, f"v-mem0-{k}-{b}", fmt(v))

    stats = load("dataset_stats")
    for name, entry in stats.items():
        define(lines, f"v-full-{name}", f"{entry['full_tokens']:,.0f}".replace(",", "{,}"))
        define(lines, f"v-goldtok-{name}", f"{entry['gold_tokens']:.0f}")
        define(lines, f"v-othertok-{name}", f"{entry['other_tokens']:.0f}")

    online = load("online")
    titles = {"lme": "LongMemEval-S", "locomo": "LoCoMo", "repomem": "RepoMem"}
    columns = ("static", "online", "online-ucb", "online-noisy", "online-debiased")
    rows = ["\\begin{tabular}{@{}lrrrrr@{}}\\toprule", "Dataset & Static & Online & UCB & Noisy & Debiased \\\\\\midrule"]
    for d, settings in online["summary"].items():
        cells = []
        for s in columns:
            values = settings[s]["all"]
            mean = sum(values) / len(values)
            sd = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            cells.append(f"{fmt(mean)} $\\pm$ {fmt(sd)}")
            define(lines, f"v-on-{d}-{s}", fmt(mean))
        rows.append(f"{titles[d]} & " + " & ".join(cells) + " \\\\")
    rows.append("\\bottomrule\\end{tabular}")
    (OUT / "table_online.tex").write_text("\n".join(rows))

    synthetic = load("synthetic")
    for policy, slope in synthetic["loglog_slope"].items():
        define(lines, f"v-slope-{policy}", f"{slope:.2f}")
    for policy, runs in synthetic["regret"].items():
        define(lines, f"v-regret-{policy}", f"{sum(run[-1] for run in runs) / len(runs):.0f}")
    ratio = synthetic["certificate_ratio"]
    define(lines, "v-certmean", fmt(ratio["mean"]))
    define(lines, "v-certmin", fmt(ratio["min"]))
    define(lines, "v-certn", ratio["n"])

    consolidation = load("consolidation")
    rows = ["\\begin{tabular}{@{}lrrrr@{}}\\toprule", "Dataset, $\\tau$, rate & Clusters & Sources & Tokens saved & Recall@1024 \\\\\\midrule"]
    for key, entry in consolidation["results"].items():
        dataset, threshold, rate = key.split("|")
        label = "baseline" if threshold == "None" else f"$\\tau={threshold}$, $R={rate}$"
        rows.append(f"{ {'locomo': 'LoCoMo', 'lme': 'LongMemEval-S'}[dataset]}, {label} & {entry['clusters']} & {entry['sources']} & {100 * entry['saving']:.1f}\\% & {fmt(entry['recall'])} \\\\")
        define(lines, f"v-con-{dataset}-{threshold}-{rate}", fmt(entry["recall"]))
        define(lines, f"v-consave-{dataset}-{threshold}-{rate}", f"{100 * entry['saving']:.1f}")
    rows.append("\\bottomrule\\end{tabular}")
    (OUT / "table_consolidation.tex").write_text("\n".join(rows))

    forgetting = load("forgetting")
    rows = ["\\begin{tabular}{@{}lrrr@{}}\\toprule", "Policy & Active items & Active tokens & Recall@1024 \\\\\\midrule"]
    for name, entry in forgetting["results"].items():
        rows.append(f"{name} & {entry['final_active']:.0f} & {entry['final_tokens']:.0f} & {fmt(entry['recall'])} \\\\")
        define(lines, f"v-fg-{name}", fmt(entry["recall"]))
        define(lines, f"v-fgt-{name}", f"{entry['final_tokens']:.0f}")
    define(lines, "v-gapdays", f"{forgetting['meta']['session_days']:.0f}")
    rows.append("\\bottomrule\\end{tabular}")
    (OUT / "table_forgetting.tex").write_text("\n".join(rows))

    latency = load("scale")
    for key, entry in latency["results"].items():
        define(lines, f"v-p50-{key}", f"{entry['p50_ms']:.1f}")
        define(lines, f"v-p95-{key}", f"{entry['p95_ms']:.1f}")
        define(lines, f"v-wr-{key}", f"{entry['write_ms']:.2f}")
    define(lines, "v-encode", f"{latency['query_encode_ms']:.0f}")

    (OUT / "macros.tex").write_text("\n".join(lines) + "\n")
    update_readme(data, tests, {n: data[n] for n in STATIC}, latency["results"])


if __name__ == "__main__":
    sys.exit(main())
