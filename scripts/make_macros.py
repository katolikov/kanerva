"""Generate paper/generated/*.tex from results/*.json, so that the paper cannot disagree with the results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "generated"
BUDGETS = (128, 256, 512, 1024, 2048, 4096)
DATASETS = {"lme": "LongMemEval-S", "locomo": "LoCoMo", "repomem-keyed": "RepoMem, keys", "repomem-plain": "RepoMem, no keys",
            "repomem-holdout": "RepoMem holdout, keys", "repomem-holdout-plain": "RepoMem holdout, no keys"}
METHODS = {
    "kanerva": "\\kanerva", "kanerva-ce": "\\kanerva + reranker", "kanerva-first": "\\kanerva, first run", "kanerva-lex": "\\kanerva, lexical", "rerank": "Hybrid + reranker", "hybrid": "Hybrid RRF",
    "hybrid-live": "Hybrid, live only", "hybrid-dedup": "Hybrid, newest of twins", "bm25": "BM25", "dense": "Dense", "mmr": "MMR",
    "facility": "Facility location", "gen-agents": "Generative Agents", "recency": "Recency",
}
ABLATIONS = {"-density": "no cost density", "-coverage": "no facets", "-redundancy": "no redundancy term", "-validity": "no validity",
             "-supersession": "no soft supersession", "-activation": "no activation"}


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def define(lines: list[str], key: str, value) -> None:
    lines.append(f"\\expandafter\\def\\csname {key}\\endcsname{{{value}}}")


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def signed(x: float, digits: int = 3) -> str:
    return f"{x:+.{digits}f}"


def pvalue(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def recall_table(comparison: dict) -> str:
    rows = ["\\begin{tabular}{@{}l" + "r" * len(BUDGETS) + "@{}}\\toprule", "Method & " + " & ".join(str(b) for b in BUDGETS) + " \\\\\\midrule"]
    for name, title in DATASETS.items():
        if name not in comparison:
            continue
        means = comparison[name]["means"]
        rows.append(f"\\multicolumn{{{len(BUDGETS) + 1}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        best = {b: max(means[f"{m}@{b}"] for m in METHODS if f"{m}@{b}" in means) for b in BUDGETS}
        for method, label in METHODS.items():
            if f"{method}@128" not in means or (method in ("hybrid-live", "hybrid-dedup") and not name.startswith("repomem")) or (method == "kanerva-first" and name not in ("repomem-keyed", "repomem-holdout")):
                continue
            cells = []
            for b in BUDGETS:
                v = means[f"{method}@{b}"]
                text = fmt(v)
                cells.append(f"\\textbf{{{text}}}" if abs(v - best[b]) < 5e-4 else text)
            rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    rows[-1] = "\\bottomrule\\end{tabular}"
    return "\n".join(rows)


def diff_table(comparison: dict, key: str = "comparisons") -> str:
    rows = ["\\begin{tabular}{@{}lllrrrrr@{}}\\toprule", "Dataset & Rival & $B$ & $\\Delta$ recall & Query CI & Unit CI & Units $+$ & $p$ \\\\\\midrule"]
    for name, title in DATASETS.items():
        if name not in comparison:
            continue
        tests = comparison[name].get(key, [])
        if not tests:
            continue
        rivals = {}
        for b in (256, 1024, 4096):
            candidates = [t for t in tests if t["budget"] == b]
            if not candidates:
                continue
            rivals[b] = max(candidates, key=lambda t: t["mean_b"])
        for b, t in rivals.items():
            cluster = "---" if not t["cluster_ci"] else f"[{signed(t['cluster_ci'][0])}, {signed(t['cluster_ci'][1])}]"
            rows.append(f"{title} & {METHODS[t['other']]} & {b} & {signed(t['diff'])} & [{signed(t['ci'][0])}, {signed(t['ci'][1])}] & {cluster} & {t['units_positive'] or '---'} & {pvalue(t['p_holm'])} \\\\")
        rows.append("\\addlinespace")
    rows[-1] = "\\bottomrule\\end{tabular}"
    return "\n".join(rows)


def ablation_table(comparison: dict) -> str:
    names = [n for n in DATASETS if n in comparison]
    budgets = (256, 1024)
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{DATASETS[n]}}}" for n in names)
    rows = ["\\begin{tabular}{@{}l" + "rr" * len(names) + "@{}}\\toprule", f"Removed & {head} \\\\", " & " + " & ".join(" & ".join(str(b) for b in budgets) for _ in names) + " \\\\\\midrule"]
    for ablation, label in ABLATIONS.items():
        cells = []
        for n in names:
            tests = {t["budget"]: t for t in comparison[n]["ablations"] if t["other"] == ablation}
            for b in budgets:
                t = tests[b]
                mark = "$^{*}$" if t["p_holm"] < 0.05 else ""
                cells.append(signed(t["diff"]) + mark)
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    rows.append("\\bottomrule\\end{tabular}")
    return "\n".join(rows)


def staleness_table(comparison: dict) -> str:
    methods = ("kanerva", "kanerva-lex", "-supersession", "-validity", "hybrid-live", "hybrid-dedup", "hybrid-newest", "hybrid", "dense", "bm25", "rerank")
    labels = {**METHODS, "-validity": "\\kanerva, no validity", "-supersession": "\\kanerva, no soft supersession", "hybrid-newest": "Hybrid, newest first"}
    names = [n for n in ("repomem-keyed", "repomem-plain", "repomem-holdout-plain", "lme-updates") if n in comparison]
    titles = {"repomem-keyed": "RepoMem, keys", "repomem-plain": "RepoMem, no keys", "repomem-holdout-plain": "Holdout, no keys", "lme-updates": "LME updates"}
    rows = ["\\begin{tabular}{@{}l" + "rr" * len(names) + "@{}}\\toprule"]
    rows.append("Method & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{titles[n]}}}" for n in names) + " \\\\")
    rows.append(" & " + " & ".join("inject & error" for _ in names) + " \\\\\\midrule")
    for m in methods:
        cells = []
        for n in names:
            inject = comparison[n]["inject_changed"].get(f"{m}@1024")
            error = comparison[n]["stale_changed"].get(f"{m}@1024")
            cells.extend("---" if v is None else fmt(v) for v in (inject, error))
        rows.append(f"{labels[m]} & " + " & ".join(cells) + " \\\\")
    rows.append("\\bottomrule\\end{tabular}")
    return "\n".join(rows)


def category_table(comparison: dict) -> str:
    rows = ["\\begin{tabular}{@{}lrrrrrr@{}}\\toprule", "Category & $n$ & \\kanerva & + reranker & Hybrid & Rerank & BM25 \\\\\\midrule"]
    for name in ("lme", "repomem-keyed"):
        if name not in comparison:
            continue
        for category, entry in comparison[name]["by_category_1024"].items():
            cells = [fmt(entry[m]) if m in entry else "---" for m in ("kanerva", "kanerva-ce", "hybrid", "rerank", "bm25")]
            rows.append(f"{category.replace('-', ' ')} & {entry['n']} & " + " & ".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    rows[-1] = "\\bottomrule\\end{tabular}"
    return "\n".join(rows)


def update_readme(comparison: dict) -> None:
    rows = ["| Dataset | Budget | Kanerva | Kanerva + reranker | Hybrid RRF | Hybrid + reranker | BM25 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("lme", "locomo", "repomem-plain", "repomem-holdout-plain"):
        if name not in comparison:
            continue
        means = comparison[name]["means"]
        for b in (256, 1024, 4096):
            cells = [fmt(means[f"{m}@{b}"]) if f"{m}@{b}" in means else "n/a" for m in ("kanerva", "kanerva-ce", "hybrid", "rerank", "bm25")]
            rows.append(f"| {DATASETS[name]} | {b} | " + " | ".join(cells) + " |")
    table = "\n".join(rows)
    stale = []
    for name in ("repomem-plain", "repomem-keyed", "lme-updates"):
        if name in comparison:
            s = comparison[name]["inject_changed"]
            stale.append(f"{DATASETS.get(name, name)}: {fmt(s['kanerva@1024'])} for Kanerva, {fmt(s['hybrid@1024'])} for hybrid.")
    text = (
        "Evidence recall on test splits, with no LLM in the evaluation. Larger is better.\n\n" + table
        + "\n\nShare of changed-fact queries where an outdated version is injected, at 1024 tokens. " + " ".join(stale)
        + "\n\nA cross-encoder reranker beats plain Kanerva at small budgets, and the reranker variant was designed after seeing "
        "those test results, so it needs confirming on fresh data. Hybrid retrieval beats Kanerva at large budgets on LoCoMo, and on "
        "recall on RepoMem without keys, where Kanerva trades a little recall for far fewer stale injections. "
        "The paper reports every such case, with confidence intervals."
    )
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
        "\\newcommand{\\T}[3]{\\csname t-#1-#2-#3\\endcsname}",
        "\\newcommand{\\St}[3]{\\csname s-#1-#2-#3\\endcsname}",
        "\\newcommand{\\D}[3]{\\csname d-#1-#2-#3\\endcsname}",
        "\\newcommand{\\CI}[3]{\\csname ci-#1-#2-#3\\endcsname}",
        "\\newcommand{\\V}[1]{\\csname v-#1\\endcsname}",
        "\\newcommand{\\Err}[3]{\\csname e-#1-#2-#3\\endcsname}",
        "\\newcommand{\\Inj}[3]{\\csname i-#1-#2-#3\\endcsname}",
    ]
    comparison = load("comparison") or {}
    for name, entry in comparison.items():
        for key, value in entry["means"].items():
            m, b = key.split("@")
            define(lines, f"r-{name}-{m}-{b}", fmt(value))
        for key, value in entry["tokens"].items():
            m, b = key.split("@")
            define(lines, f"t-{name}-{m}-{b}", f"{value:.0f}")
        for key, value in entry["stale_error"].items():
            m, b = key.split("@")
            define(lines, f"s-{name}-{m}-{b}", fmt(value))
        for t in entry["comparisons"] + entry["ablations"]:
            define(lines, f"d-{name}-{t['other']}-{t['budget']}", signed(t["diff"]))
            define(lines, f"ci-{name}-{t['other']}-{t['budget']}", f"[{signed(t['ci'][0])}, {signed(t['ci'][1])}]")
        define(lines, f"v-n-{name}", entry["comparisons"][0]["n"])
        for other, p in entry.get("pareto_ce", {}).items():
            define(lines, f"v-cewin-{name}-{other}", p["wins"])
            define(lines, f"v-celoss-{name}-{other}", p["losses"])
        if "latency_rerank_ms" in entry:
            define(lines, f"v-latce-{name}", f"{entry['latency_rerank_ms']:.0f}")
        for other, p in entry["pareto"].items():
            define(lines, f"v-pwin-{name}-{other}", p["wins"])
            define(lines, f"v-ploss-{name}-{other}", p["losses"])

    for name, entry in comparison.items():
        for key, value in entry["stale_changed"].items():
            m, b = key.split("@")
            define(lines, f"e-{name}-{m}-{b}", fmt(value))
            define(lines, f"i-{name}-{m}-{b}", fmt(entry["inject_changed"][key]))
        define(lines, f"v-nchanged-{name}", entry["changed_n"])
    if comparison:
        gains = [-t["diff"] for t in comparison["locomo"]["ablations"] if t["other"] == "-density"]
        define(lines, "v-dens-locomo", fmt(max(gains)))

    systems = load("systems_mem0")
    raw_lme = RESULTS / "raw" / "retrieval_lme.json.gz"
    if systems and raw_lme.exists():
        import gzip

        qids = {r["qid"] for r in systems["rows"]}
        records = [r for r in json.load(gzip.open(raw_lme, "rt")) if r["qid"] in qids]
        for b in (512, 1024):
            mem0 = [r["recall"] for r in systems["rows"] if r["budget"] == b]
            ours = [r["recall"] for r in records if r["budget"] == b and r["method"] == "kanerva"]
            hyb = [r["recall"] for r in records if r["budget"] == b and r["method"] == "hybrid"]
            define(lines, f"v-mem0-{b}", fmt(float(np.mean(mem0))))
            define(lines, f"v-mem0ours-{b}", fmt(float(np.mean(ours))))
            define(lines, f"v-mem0hyb-{b}", fmt(float(np.mean(hyb))))
        define(lines, "v-mem0n", systems["meta"]["questions"])
        define(lines, "v-mem0add", f"{systems['meta']['add_ms_per_turn']:.0f}")
        define(lines, "v-mem0search", f"{systems['meta']['search_ms']:.0f}")

    for name in ("lme", "locomo", "repomem-plain"):
        small = load(f"m2v_{name}")
        if small:
            for method in ("kanerva", "hybrid", "dense"):
                define(lines, f"v-m2v-{name}-{method}", fmt(small["summary"][f"{method}@1024"]["recall"]))

    stats = load("dataset_stats")
    if stats:
        for name, entry in stats.items():
            define(lines, f"v-full-{name}", f"{entry['full_tokens']:,.0f}".replace(",", "{,}"))
            define(lines, f"v-goldtok-{name}", f"{entry['gold_tokens']:.0f}")
            define(lines, f"v-othertok-{name}", f"{entry['other_tokens']:.0f}")

    online = load("online")
    online_rows = ""
    if online:
        titles = {"lme": "LongMemEval-S", "locomo": "LoCoMo", "repomem": "RepoMem"}
        columns = ("static", "online", "online-ucb", "online-noisy", "online-debiased", "cold-static", "cold-online")
        rows = ["\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule", "Dataset & Static & Online & UCB & Noisy & Debiased & Flat static & Flat online \\\\\\midrule"]
        for d, settings in online["summary"].items():
            cells = []
            for s in columns:
                if s not in settings:
                    cells.append("---")
                    continue
                v = float(np.mean(settings[s]["second_half"]))
                sd = float(np.std(settings[s]["second_half"]))
                cells.append(f"{fmt(v)} $\\pm$ {fmt(sd)}")
                define(lines, f"v-on-{d}-{s}", fmt(v))
            rows.append(f"{titles[d]} & " + " & ".join(cells) + " \\\\")
        rows.append("\\bottomrule\\end{tabular}")
        online_rows = "\n".join(rows)

    synthetic = load("synthetic")
    if synthetic:
        for policy, slope in synthetic["loglog_slope"].items():
            define(lines, f"v-slope-{policy}", f"{slope:.2f}")
        final = {p: float(np.mean([run[-1] for run in runs])) for p, runs in synthetic["regret"].items()}
        for policy, value in final.items():
            define(lines, f"v-regret-{policy}", f"{value:.0f}")
        ratio = synthetic["certificate_ratio"]
        define(lines, "v-certmean", fmt(ratio["mean"]))
        define(lines, "v-certmin", fmt(ratio["min"]))
        define(lines, "v-certn", ratio["n"])

    consolidation = load("consolidation")
    if consolidation:
        rows = ["\\begin{tabular}{@{}lrrrrr@{}}\\toprule", "Dataset, $\\tau$, rate & Clusters & Sources & Tokens saved & Recall@1024 \\\\\\midrule"]
        for key, entry in consolidation["results"].items():
            dataset, threshold, rate = key.split("|")
            label = "baseline" if threshold == "None" else f"$\\tau={threshold}$, $R={rate}$"
            rows.append(f"{ {'locomo': 'LoCoMo', 'lme': 'LongMemEval-S'}[dataset]}, {label} & {entry['clusters']} & {entry['sources']} & {100 * entry['saving']:.1f}\\% & {fmt(entry['recall'])} \\\\")
            define(lines, f"v-con-{dataset}-{threshold}-{rate}", fmt(entry["recall"]))
            define(lines, f"v-consave-{dataset}-{threshold}-{rate}", f"{100 * entry['saving']:.1f}")
        rows.append("\\bottomrule\\end{tabular}")
        (OUT / "table_consolidation.tex").write_text("\n".join(rows))

    forgetting = load("forgetting")
    if forgetting:
        rows = ["\\begin{tabular}{@{}lrrr@{}}\\toprule", "Policy & Active items & Active tokens & Recall@1024 \\\\\\midrule"]
        for name, entry in forgetting["results"].items():
            rows.append(f"{name} & {entry['final_active']:.0f} & {entry['final_tokens']:.0f} & {fmt(entry['recall'])} \\\\")
            define(lines, f"v-fg-{name}", fmt(entry["recall"]))
            define(lines, f"v-fgt-{name}", f"{entry['final_tokens']:.0f}")
        define(lines, "v-gapdays", f"{forgetting['meta']['session_days']:.0f}")
        rows.append("\\bottomrule\\end{tabular}")
        (OUT / "table_forgetting.tex").write_text("\n".join(rows))

    multires = load("multires")
    if multires:
        for key, entry in multires["results"].items():
            define(lines, f"v-mr-{key}", fmt(entry["recall"]))
            define(lines, f"v-mrcut-{key}", f"{entry['cut']:.1f}")

    scale = load("scale")
    if scale:
        for key, entry in scale["results"].items():
            define(lines, f"v-p50-{key}", f"{entry['p50_ms']:.1f}")
            define(lines, f"v-p95-{key}", f"{entry['p95_ms']:.1f}")
            define(lines, f"v-wr-{key}", f"{entry['write_ms']:.2f}")
        define(lines, "v-encode", f"{scale['query_encode_ms']:.0f}")

    (OUT / "macros.tex").write_text("\n".join(lines) + "\n")
    if comparison:
        (OUT / "table_recall.tex").write_text(recall_table(comparison))
        (OUT / "table_diff.tex").write_text(diff_table(comparison))
        (OUT / "table_diff_ce.tex").write_text(diff_table(comparison, "comparisons_ce"))
        (OUT / "table_ablation.tex").write_text(ablation_table(comparison))
        (OUT / "table_stale.tex").write_text(staleness_table(comparison))
        (OUT / "table_category.tex").write_text(category_table(comparison))
    if online_rows:
        (OUT / "table_online.tex").write_text(online_rows)
    if comparison:
        update_readme(comparison)


if __name__ == "__main__":
    sys.exit(main())
