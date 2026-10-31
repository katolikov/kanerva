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
NEW = {"repomem-new": "RepoMem, keys", "repomem-new-plain": "RepoMem, no keys"}
FINAL = {"repomem-final": "RepoMem, keys", "repomem-final-plain": "RepoMem, no keys"}
METHODS = {
    "kanerva-ce": "\\kanerva + reranker", "kanerva": "\\kanerva", "kanerva-events": "\\kanerva, earlier rule", "kanerva-lex": "\\kanerva, lexical", "kanerva-r1": "First version",
    "kanerva-0.4": "\\kanerva 0.4.1", "kanerva-lex-0.4": "\\kanerva 0.4.1, lexical",
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
NEW_PAIRS = {"kanerva vs kanerva-events": "\\kanerva vs earlier rule", "kanerva vs hybrid": "\\kanerva vs hybrid", "kanerva vs rerank": "\\kanerva vs reranker",
             "kanerva vs hybrid-live": "\\kanerva vs hybrid, live", "kanerva-lex vs bm25": "Lexical vs BM25"}
FINAL_PAIRS = {"kanerva vs hybrid": "\\kanerva vs hybrid", "kanerva vs rerank": "\\kanerva vs reranker", "kanerva vs hybrid-live": "\\kanerva vs hybrid, live",
               "kanerva vs kanerva-0.4": "\\kanerva vs 0.4.1", "kanerva vs kanerva-any-twin": "\\kanerva vs any twin", "kanerva-lex vs bm25": "Lexical vs BM25",
               "kanerva-lex vs kanerva-lex-0.4": "Lexical vs 0.4.1", "kanerva-lex vs kanerva-lex-any-twin": "Lexical vs any twin"}
FINAL_SKIP = ("gen-agents", "facility", "recency", "mmr", "dense", "hybrid-dedup")
FINAL_STALE = ("kanerva", "kanerva-lex", "kanerva-0.4", "kanerva-lex-0.4", "hybrid-dedup", "hybrid-live", "hybrid", "rerank", "bm25")
LANGUAGE_METHODS = {"kanerva": "\\kanerva", "hybrid": "Hybrid", "rerank": "Reranker", "kanerva-lex": "Lexical", "bm25": "BM25", "kanerva-0.4": "0.4.1"}
GATE_SETS = {"lme": "LongMemEval-S", "locomo": "LoCoMo", "beam-dev": "BEAM-1M", "repomem": "RepoMem dev", "repomem-holdout": "RepoMem holdout",
             "repomem-fresh": "RepoMem first test", "repomem-new": "RepoMem second test", "repomem-other": "RepoMem beyond Python"}
FILE_SETS = {"repomem-final": "Third test", "repomem-other": "Beyond Python", "repomem-new": "Second test", "repomem-fresh": "First test",
             "repomem-holdout": "Holdout", "repomem": "RepoMem dev"}
WRITE_SETS = {"repomem-final": "RepoMem, third test", "repomem-new": "RepoMem, second test", "repomem-fresh": "RepoMem, first test", "lme-updates": "LongMemEval updates"}
ONLINE_COLUMNS = {"static": "Static", "online": "Bandit", "online-noisy": "Bandit, noisy", "lift": "Lift", "lift-noisy": "Lift, noisy"}
ANSWER_SETS = {"locomo": "LoCoMo", "lme": "LongMemEval-S", "beam-dev": "BEAM-1M", "repomem-final": "RepoMem, third test"}
ANSWER_METHODS = {"kanerva": "\\kanerva", "kanerva-lex": "\\kanerva, lexical", "kanerva-0.4": "\\kanerva 0.4.1", "kanerva-lex-0.4": "\\kanerva 0.4.1, lexical", "hybrid": "Hybrid RRF", "bm25": "BM25"}
ANSWER_PAIRS = {"kanerva vs hybrid": "\\kanerva vs hybrid", "kanerva vs bm25": "\\kanerva vs BM25", "kanerva-lex vs bm25": "Lexical vs BM25",
                "kanerva vs kanerva-0.4": "\\kanerva vs 0.4.1", "kanerva-lex vs kanerva-lex-0.4": "Lexical vs 0.4.1"}
ANSWER_BUDGETS = (256, 512, 1024)
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


def percent(x: float) -> str:
    return f"{100 * x:.1f}"


def points(x: float) -> str:
    return f"{100 * x:+.1f}"


def sci(x: float) -> str:
    exponent = len(str(int(x))) - 1
    return f"${x / 10 ** exponent:.1f}\\times10^{{{exponent}}}$"


def pvalue(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def mean_over_budgets(summary: dict, method: str, metric: str = "recall") -> float:
    return sum(summary[f"{method}@{b}"][metric] for b in BUDGETS) / len(BUDGETS)


def recall_table(data: dict, titles: dict = FRESH, skip: tuple = ()) -> str:
    rows = ["\\begin{tabular}{@{}l" + "r" * (len(BUDGETS) + 1) + "@{}}\\toprule", "Method & " + " & ".join(map(str, BUDGETS)) + " & mean \\\\\\midrule"]
    for name, title in titles.items():
        summary = data[name]["summary"]
        rows.append(f"\\multicolumn{{{len(BUDGETS) + 2}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        methods = [m for m in METHODS if f"{m}@128" in summary and m not in skip]
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


def tests_table(tests: dict, pairs: dict, budgets: tuple[str, ...], titles: dict = FRESH) -> str:
    rows = ["\\begin{tabular}{@{}lllrrrr@{}}\\toprule", "Dataset & Comparison & $B$ & $\\Delta$ recall & Unit CI & Units $+$ & $p$ \\\\\\midrule"]
    for name, title in titles.items():
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


def final_stale_table(tests: dict) -> str:
    rows = ["\\begin{tabular}{@{}lrrrr@{}}\\toprule", "Method & \\multicolumn{2}{c}{Keys} & \\multicolumn{2}{c}{No keys} \\\\", " & inject & error & inject & error \\\\\\midrule"]
    for m in FINAL_STALE:
        cells = []
        for name in FINAL:
            row = tests[name]["stale_on_changed_facts"][m]
            cells += [fmt(row["stale_any"]), fmt(row["stale_error"])]
        label = "Hybrid, newest of twins" if m == "hybrid-dedup" else METHODS[m]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    rows.append("\\bottomrule\\end{tabular}")
    return "\n".join(rows)


def languages_table(languages: dict) -> str:
    rows = ["\\begin{tabular}{@{}ll" + "r" * len(LANGUAGE_METHODS) + "@{}}\\toprule", "Language & Track & " + " & ".join(LANGUAGE_METHODS.values()) + " \\\\\\midrule"]
    for language, entry in languages.items():
        for track in ("fact", "task"):
            if f"{track}/kanerva" not in entry:
                continue
            means = [entry[f"{track}/{m}"][-1] for m in LANGUAGE_METHODS]
            best = max(means)
            cells = [f"\\textbf{{{fmt(v)}}}" if abs(v - best) < 5e-4 else fmt(v) for v in means]
            rows.append(f"{language} & {track} & " + " & ".join(cells) + " \\\\")
            language = ""
    rows.append("\\bottomrule\\end{tabular}")
    return "\n".join(rows)


def gate_table(spent: dict, final: dict) -> str:
    head = ["\\begin{tabular}{@{}lrrrrrrrr@{}}\\toprule", " & \\multicolumn{4}{c}{Own questions} & \\multicolumn{2}{c}{Same kind, foreign} & \\multicolumn{2}{c}{Other kind, foreign} \\\\",
            "Set & $n$ & recall & with gate & spoke & without & with gate & without & with gate \\\\\\midrule"]
    rows = list(head)
    for config, title in (("full", "With vectors"), ("lexical", "Lexical")):
        rows.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        entries = [("Third test", final[config])] + [(label, spent["sets"][name][config]) for name, label in GATE_SETS.items()]
        for label, entry in entries:
            before, after = entry["without"], entry["with"]
            rows.append(f"{label} & {before['own']['queries']} & {fmt(before['own']['recall'])} & {fmt(after['own']['recall'])} & {fmt(after['own']['spoke'])} & "
                        f"{fmt(before['same']['spoke'])} & {fmt(after['same']['spoke'])} & {fmt(before['other']['spoke'])} & {fmt(after['other']['spoke'])} \\\\")
        rows.append("\\addlinespace")
    rows[-1] = "\\bottomrule\\end{tabular}"
    return "\n".join(rows)


def files_table(sets: dict) -> str:
    rows = ["\\begin{tabular}{@{}llrrrrrr@{}}\\toprule", "Set & Session & Prompt & Same tokens & With notes & Precision & Better & Worse \\\\\\midrule"]
    for name, label in FILE_SETS.items():
        for session, title in (("one", "this task"), ("previous", "last task")):
            row = sets[name][f"{session}/full"]
            move = row["with_files_vs_wider"]
            rows.append(f"{label} & {title} & {fmt(row['prompt'])} & {fmt(row['wider'])} & {fmt(row['with_files'])} & {fmt(row['file_precision'])} & {move['up']} & {move['down']} \\\\")
    rows.append("\\bottomrule\\end{tabular}")
    return "\n".join(rows)


def write_path_table(report: dict) -> str:
    rows = ["\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule", "Set & Changed & First & Listed & List on others & Stale & Stale, oracle & Tokens \\\\\\midrule"]
    for arm, title in (("vectors", "With vectors"), ("lexical", "Lexical")):
        rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for name, label in WRITE_SETS.items():
            entry = report[arm][name]
            similar, stale = entry["similar"], entry["stale_on_changed_facts"]
            changed = f"{similar['changed_notes']:,}".replace(",", "{,}")
            rows.append(f"{label} & {changed} & {fmt(similar['top1'])} & {fmt(similar['top3'])} & {fmt(similar['shown_on_other_notes'])} & "
                        f"{fmt(stale['current_rule']['stale_any'])} & {fmt(stale['oracle_simulated']['stale_any'])} & {similar['list_tokens_when_shown']:.0f} \\\\")
        rows.append("\\addlinespace")
    rows[-1] = "\\bottomrule\\end{tabular}"
    return "\n".join(rows)


def answers_table(sets: dict) -> str:
    columns = len(ANSWER_SETS) * len(ANSWER_BUDGETS)
    head = " & ".join(f"\\multicolumn{{{len(ANSWER_BUDGETS)}}}{{c}}{{{title}}}" for title in ANSWER_SETS.values())
    rows = ["\\begin{tabular}{@{}l" + "r" * columns + "@{}}\\toprule", f"Method & {head} \\\\", " & " + " & ".join(map(str, ANSWER_BUDGETS * len(ANSWER_SETS))) + " \\\\\\midrule"]
    for method, label in ANSWER_METHODS.items():
        cells = [fmt(sets[name]["answers"][f"{method}@{b}"]["contain"]) for name in ANSWER_SETS for b in ANSWER_BUDGETS]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    rows.append("\\addlinespace")
    gold = [f"\\multicolumn{{{len(ANSWER_BUDGETS)}}}{{c}}{{{fmt(sets[name]['answers']['gold@1024']['contain'])}}}" for name in ANSWER_SETS]
    rows.append("Labelled evidence & " + " & ".join(gold) + " \\\\")
    rows.append("\\bottomrule\\end{tabular}")
    return "\n".join(rows)


def answer_tests_table(sets: dict) -> str:
    rows = ["\\begin{tabular}{@{}llrrrr@{}}\\toprule", "Dataset & Comparison & $\\Delta$ & Interval & Units $+$ & $p$ \\\\\\midrule"]
    for name, title in ANSWER_SETS.items():
        for pair, label in ANSWER_PAIRS.items():
            t = sets[name]["tests"][pair]["contain"]["mean"]
            low, high = t["cluster_ci"] or t["ci"]
            units = t["units_positive"] or "---"
            rows.append(f"{title} & {label} & {signed(t['diff'])} & [{signed(low)}, {signed(high)}] & {units} & {pvalue(t['p'])} \\\\")
        rows.append("\\addlinespace")
    rows[-1] = "\\bottomrule\\end{tabular}"
    return "\n".join(rows)


def answer_stale_table(stale: dict) -> str:
    rows = ["\\begin{tabular}{@{}lrrrrrr@{}}\\toprule", "Method & \\multicolumn{3}{c}{Old value answered} & \\multicolumn{3}{c}{Stale note injected} \\\\",
            " & " + " & ".join(map(str, ANSWER_BUDGETS * 2)) + " \\\\\\midrule"]
    for method, label in ANSWER_METHODS.items():
        cells = [percent(stale[f"{method}@{b}"][field]) for field in ("old", "injected") for b in ANSWER_BUDGETS]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    rows.append("\\bottomrule\\end{tabular}")
    return "\n".join(rows)


def every_test(entry: dict):
    """Each test in a nested block of results/answers.json: a dict that has a p value."""
    if "p" in entry:
        yield entry
    else:
        for value in entry.values():
            if isinstance(value, dict):
                yield from every_test(value)


C_METHODS = {"kanerva": "\\kanerva", "kanerva-ce": "\\kanerva + reranker", "kanerva-lex": "\\kanerva, lexical", "kanerva-0.4": "\\kanerva 0.4.1", "rerank": "Hybrid + reranker",
             "hybrid": "Hybrid RRF", "hybrid-dedup": "Hybrid, newest of twins", "bm25": "BM25"}
C_PAIRS = ("kanerva vs hybrid", "kanerva vs rerank", "kanerva vs kanerva-0.4", "kanerva-lex vs bm25", "kanerva-lex vs kanerva-lex-0.4")


def c_macros(lines: list[str]) -> None:
    """C and C++: the held-out test, the dev group it followed, file notes and the gate."""
    test = load("c-test_tests")
    dev = load("c-dev_tests")
    data = {"repomem-c-test": load("fresh_repomem-c-test"), "repomem-c-test-plain": load("fresh_repomem-c-test-plain")}
    define(lines, "v-c-queries", f"{data['repomem-c-test-plain']['meta']['queries']:,}".replace(",", "{,}"))
    for name, entry in data.items():
        for key, row in entry["summary"].items():
            method, budget = key.split("@")
            define(lines, f"v-c-r-{name}-{method}-{budget}", fmt(row["recall"]))
        for method in sorted({k.split("@")[0] for k in entry["summary"]}):
            define(lines, f"v-c-r-{name}-{method}-mean", fmt(mean_over_budgets(entry["summary"], method)))
    for name, report in (("test", test), ("dev", dev)):
        plain = report[f"repomem-c-{name}-plain"]
        keyed = report[f"repomem-c-{name}"]
        for label, entry in (("plain", plain), ("keys", keyed)):
            for pair in C_PAIRS:
                key = pair.replace(" ", "_")
                for budget, t in entry[pair].items():
                    define(lines, f"v-cd-{name}-{label}-{key}-{budget}", signed(t["diff"]))
                    define(lines, f"v-cci-{name}-{label}-{key}-{budget}", f"[{signed(t['cluster_ci'][0])}, {signed(t['cluster_ci'][1])}]")
                    define(lines, f"v-cpos-{name}-{label}-{key}-{budget}", t["units_positive"])
                    define(lines, f"v-cp-{name}-{label}-{key}-{budget}", pvalue(t.get("p_holm", t["p"])))
            for method, row in entry["stale_on_changed_facts"].items():
                define(lines, f"v-cst-{name}-{label}-{method}", percent(row["stale_any"]))
            for key, value in entry["tracks"].items():
                track, rest = key.split("/")
                method, budget = rest.split("@")
                define(lines, f"v-ctrack-{name}-{label}-{track}-{method}-{budget}", fmt(value))
    for language, entry in test["languages"].items():
        safe = language.replace("+", "p")
        for key, values in entry.items():
            if key == "queries":
                continue
            track, method = key.split("/")
            define(lines, f"v-clang-{safe}-{track}-{method}", fmt(values[-1]))
    files = load("file_recall_c")["repomem-c-test"]
    for session in ("one", "previous"):
        for config in ("full", "lexical"):
            row = files[f"{session}/{config}"]
            prefix = f"v-cf-{session}-{config}"
            for field in ("prompt", "wider", "with_files"):
                define(lines, f"{prefix}-{field}", fmt(row[field]))
            move = row["with_files_vs_wider"]
            define(lines, f"{prefix}-gain", signed(move["mean"]))
            define(lines, f"{prefix}-up", f"{move['up']}/{move['repositories']}")
            define(lines, f"{prefix}-down", f"{move['down']}/{move['repositories']}")
    gate = load("abstention_c")
    for config in ("full", "lexical"):
        for arm in ("without", "with"):
            for kind in ("own", "same", "other"):
                define(lines, f"v-cg-{config}-{arm}-{kind}", fmt(gate[config][arm][kind]["spoke"]))
            define(lines, f"v-cg-{config}-{arm}-recall", fmt(gate[config][arm]["own"]["recall"]))
    rows = ["\\begin{tabular}{@{}l" + "r" * (len(BUDGETS) + 1) + "@{}}\\toprule", "Method & " + " & ".join(map(str, BUDGETS)) + " & mean \\\\\\midrule"]
    summary = data["repomem-c-test-plain"]["summary"]
    methods = [m for m in C_METHODS if f"{m}@128" in summary]
    best = {b: max(summary[f"{m}@{b}"]["recall"] for m in methods) for b in BUDGETS}
    best_mean = max(mean_over_budgets(summary, m) for m in methods)
    for m in methods:
        cells = [f"\\textbf{{{fmt(summary[f'{m}@{b}']['recall'])}}}" if abs(summary[f"{m}@{b}"]["recall"] - best[b]) < 5e-4 else fmt(summary[f"{m}@{b}"]["recall"]) for b in BUDGETS]
        mean = mean_over_budgets(summary, m)
        cells.append(f"\\textbf{{{fmt(mean)}}}" if abs(mean - best_mean) < 5e-4 else fmt(mean))
        rows.append(f"{C_METHODS[m]} & " + " & ".join(cells) + " \\\\")
    rows.append("\\bottomrule\\end{tabular}")
    (OUT / "table_c.tex").write_text("\n".join(rows))


def answer_macros(lines: list[str], sets: dict) -> None:
    tests = [t for s in sets.values() for block in ("tests", "stale_tests", "noise_tests", "gate_tests") for t in every_test(s.get(block, {}))]
    define(lines, "v-qa-tests", len(tests))
    define(lines, "v-qa-raw", sum(t["p"] < 0.05 for t in tests))
    define(lines, "v-qa-holm", sum(t.get("p_holm", 1.0) < 0.05 for t in tests))
    means = [s["tests"][pair]["contain"]["mean"] for s in sets.values() for pair in ANSWER_PAIRS]
    define(lines, "v-qa-means", len(means))
    define(lines, "v-qa-means-raw", sum(t["p"] < 0.05 for t in means))
    noise = sum(s["noise"]["kanerva@512"]["n"] for s in sets.values())
    define(lines, "v-qa-foreign", noise)
    define(lines, "v-qa-min-p", pvalue(2 / 2 ** 6))
    for name, s in sets.items():
        define(lines, f"v-qan-{name}", s["queries"])
        for key, row in s["answers"].items():
            method, budget = key.split("@")
            for field in ("contain", "f1", "recall", "unknown"):
                define(lines, f"v-qa-{name}-{method}-{budget}-{field}", fmt(row[field]))
        for pair, metrics in s["tests"].items():
            key = pair.replace(" ", "_")
            for budget, t in metrics["contain"].items():
                define(lines, f"v-qad-{name}-{key}-{budget}", points(t["diff"]))
                low, high = t["cluster_ci"] or t["ci"]
                define(lines, f"v-qaci-{name}-{key}-{budget}", f"{points(low)} to {points(high)}")
                define(lines, f"v-qap-{name}-{key}-{budget}", pvalue(t.get("p_holm", t["p"])))
                define(lines, f"v-qapos-{name}-{key}-{budget}", t["units_positive"] or "---")
        for budget in ANSWER_BUDGETS[1:2]:  # recall against answers, at the plugin budget
            for method in ("kanerva", "bm25"):
                row = s["answers"][f"{method}@{budget}"]
                define(lines, f"v-qapts-{name}-{method}-{budget}", f"{100 * row['recall']:.1f}")
            gap = s["answers"][f"kanerva@{budget}"]
            base = s["answers"][f"bm25@{budget}"]
            define(lines, f"v-qagap-{name}-recall-{budget}", points(gap["recall"] - base["recall"]))
            define(lines, f"v-qagap-{name}-contain-{budget}", points(gap["contain"] - base["contain"]))
        for method, row in s["noise"].items():
            short = method.split("@")[0]
            define(lines, f"v-qaspoke-{name}-{short}", percent(row["injected"]))
    for method in sorted({k.split("@")[0] for s in sets.values() for k in s["noise"]}):
        key = next(k for k in next(iter(sets.values()))["noise"] if k.startswith(f"{method}@"))
        define(lines, f"v-qainvented-{method}", round(sum(s["noise"][key]["invented"] * s["noise"][key]["n"] for s in sets.values())))
        define(lines, f"v-qaspoke-all-{method}", percent(sum(s["noise"][key]["injected"] for s in sets.values()) / len(sets)))
    final = sets["repomem-final"]
    define(lines, "v-qast-n", final["stale"]["kanerva@512"]["n"])
    for key, row in final["stale"].items():
        method, budget = key.split("@")
        for field in ("old", "injected"):
            define(lines, f"v-qast-{method}-{budget}-{field}", percent(row[field]))
    for pair, entry in final["stale_tests"].items():
        t = entry["mean"]
        define(lines, f"v-qastd-{pair.replace(' ', '_')}", points(t["diff"]))
        define(lines, f"v-qastp-{pair.replace(' ', '_')}", pvalue(t["p"]))
    for pair in ("kanerva+gate vs kanerva+weight", "kanerva-lex+gate vs kanerva-lex+weight"):
        for name, s in sets.items():
            t = s["gate_tests"][pair]["contain"]["512"]
            define(lines, f"v-qagate-{name}-{pair.split('+')[0]}", points(t["diff"]))
    (OUT / "table_answers.tex").write_text(answers_table(sets))
    (OUT / "table_answer_tests.tex").write_text(answer_tests_table(sets))
    (OUT / "table_answer_stale.tex").write_text(answer_stale_table(final["stale"]))


def field_macros(lines: list[str]) -> None:
    """The checks outside the benchmarks: a real agent, real prompts, and the flaws of the RepoMem facts."""
    for label, name in (("awp", "agent_write_path"), ("awpfirst", "agent_write_path_first_hint")):
        report = load(name)
        define(lines, f"v-{label}-trials", report["meta"]["trials"])
        define(lines, f"v-{label}-cost", f"{report['meta']['cost_usd']:.2f}")
        for dataset, entry in report["sets"].items():
            for kind in ("changed", "other"):
                for field in ("n", "replaced", "kept"):
                    define(lines, f"v-{label}-{dataset}-{kind}-{field}", entry[kind][field])
                define(lines, f"v-{label}-{dataset}-{kind}-fixed", entry[kind]["replaced"] + entry[kind].get("edited", 0))
                define(lines, f"v-{label}-{dataset}-{kind}-edited", entry[kind].get("edited", 0))
                low, high = entry[kind]["interval"]
                define(lines, f"v-{label}-{dataset}-{kind}-ci", f"{100 * low:.0f}--{100 * high:.0f}")
    define(lines, "v-awp-model", "Sonnet 5.5")
    near = load("agent_write_path")["own_note_cosines"]
    for kind in ("changed", "other"):
        for field in ("min", "median", "max"):
            define(lines, f"v-awp-cos-{kind}-{field}", fmt(near[kind][field], 2))
    final = load("agent_write_path")["sets"]
    define(lines, "v-awp-extra", sum(e[k]["extra"] for e in final.values() for k in ("changed", "other")))
    define(lines, "v-awp-other-n", sum(e["other"]["n"] for e in final.values()))
    define(lines, "v-awp-other-clean", sum(e["other"]["n"] - e["other"]["extra"] for e in final.values()))
    gate = load("abstention_static")["sets"]
    for arm in ("none", "minilm", "static"):
        for kind in ("same", "other"):
            define(lines, f"v-gs-{arm}-{kind}", fmt(sum(g[arm][kind]["spoke"] for g in gate.values()) / len(gate)))
        define(lines, f"v-gs-{arm}-recall", fmt(sum(g[arm]["own"]["recall"] for g in gate.values()) / len(gate)))
    real = load("real_sessions")
    for key in ("projects", "sessions", "prompts"):
        define(lines, f"v-rs-{key}", real["meta"][key])
    for arm in ("vectors", "lexical"):
        entry = real[arm]
        for group in ("all", "ru", "en"):
            row, foreign = entry[group], entry["foreign"][group]
            define(lines, f"v-rs-{arm}-{group}-n", row["prompts"])
            define(lines, f"v-rs-{arm}-{group}-known", row["known"])
            for field in ("free_spoke", "gated_spoke", "known_free_spoke", "known_gated_spoke", "known_free_hit", "known_gated_hit"):
                define(lines, f"v-rs-{arm}-{group}-{field}", percent(row[field]))
            for field in ("free_tokens", "gated_tokens"):
                define(lines, f"v-rs-{arm}-{group}-{field}", f"{row[field]:.0f}")
            for field in ("free_spoke", "gated_spoke"):
                define(lines, f"v-rs-{arm}-foreign-{group}-{field}", percent(foreign[field]))
        signal = entry["signal"]
        define(lines, f"v-rs-{arm}-fired", percent(signal["fired"]))
        define(lines, f"v-rs-{arm}-random", percent(signal["random_fired"]))
        define(lines, f"v-rs-{arm}-signal-notes", signal["notes"])
        define(lines, f"v-rs-{arm}-edit-lines", percent(entry["session_start_share_of_edit_lines"]))
    flaws = load("data_flaws")
    changed = sum(v for k, v in flaws["counts"].items() if k.endswith("/changed"))
    define(lines, "v-flaw-changed", f"{changed:,}".replace(",", "{,}"))
    define(lines, "v-flaw-moved", flaws["moved"])
    define(lines, "v-flaw-moved-p", percent(flaws["moved"] / changed))
    define(lines, "v-flaw-bogus", flaws["bogus"])
    plain = flaws["repomem-final-plain"]
    for method in ("kanerva", "kanerva-lex", "hybrid", "hybrid-dedup", "bm25"):
        define(lines, f"v-flaw-stale-{method}-all", percent(plain["stale_all"][method]))
        define(lines, f"v-flaw-stale-{method}-clean", percent(plain["stale_value_changed"][method]))
    lead = plain["leads"]["kanerva vs hybrid"]
    define(lines, "v-flaw-lead-all", signed(lead["all"]["diff"]))
    define(lines, "v-flaw-lead-clean", signed(lead["clean"]["diff"]))
    define(lines, "v-flaw-lead-pos", lead["clean"]["units_positive"])


def span(t: dict) -> str:
    return f"{points(t['cluster_ci'][0])} to {points(t['cluster_ci'][1])}"


def update_readme(data: dict, latency: dict) -> None:
    code, beam = data["repomem-final-plain"], data["beam-check"]
    columns = ("kanerva", "kanerva-lex", "kanerva-0.4", "rerank", "hybrid", "bm25")
    rows = ["| Dataset | Budget | Kanerva | Lexical | Kanerva 0.4.1 | Hybrid + reranker | Hybrid | BM25 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for title, summary in (("Code history", code["summary"]), ("Long conversations", beam["summary"])):
        for b in (256, 1024, 4096):
            values = {m: summary[f"{m}@{b}"]["recall"] for m in columns if f"{m}@{b}" in summary}
            best = max(values.values())
            cells = [("n/a" if m not in values else f"**{fmt(values[m])}**" if abs(values[m] - best) < 5e-4 else fmt(values[m])) for m in columns]
            rows.append(f"| {title} | {b} | " + " | ".join(cells) + " |")
    tests = load("final_tests")
    keyed, plain = tests["repomem-final"], tests["repomem-final-plain"]
    stale = plain["stale_on_changed_facts"]
    tracks = load("final_tracks")["results"]["repomem-final-plain"]
    languages = load("final_languages")["results"]["repomem-final-plain"]
    versus = " ".join(f"{name} {fmt(languages[name]['all/kanerva'][-1])} against {fmt(languages[name]['all/hybrid'][-1])}{mark}"
                      for name, mark in (("Python", ","), ("TypeScript", ","), ("Go", ","), ("Rust", ".")))
    check = load("beam_check")["beam"]
    refit = load("refit")["dense"]["recipes"]
    locomo = [sum(refit[r]["locomo"]) / len(refit[r]["locomo"]) for r in ("as-shipped", "all-families")]
    gate = load("abstention_final")
    full, lexical = gate["full"], gate["lexical"]
    spent = load("abstention")["sets"]["repomem-other"]["full"]
    files = load("file_recall_final")["repomem-final"]
    file_now, file_old = files["one/full"], files["previous/full"]
    writes = load("write_path")
    notes, changed = writes["vectors"]["repomem-final"]["similar"], writes["vectors"]["repomem-final"]["stale_on_changed_facts"]
    hooks = load("hook_latency")["results"]
    c_plain = load("c-test_tests")["repomem-c-test-plain"]
    c_files = load("file_recall_c")["repomem-c-test"]["one/full"]
    agent = load("agent_write_path")
    own, verbatim, code_notes = agent["sets"]["lme-own-notes"], agent["sets"]["lme-updates"], agent["sets"]["repomem-final"]
    first = load("agent_write_path_first_hint")["sets"]
    touched = sum(e[k]["extra"] for e in agent["sets"].values() for k in ("changed", "other"))
    real = load("real_sessions")
    answers = load("answers")["sets"]
    final_answers, final_stale = answers["repomem-final"], answers["repomem-final"]["stale"]
    conversations = " and ".join(f"{points(answers[name]['tests']['kanerva vs bm25']['contain']['mean']['diff'])} on {title}" for name, title in (("locomo", "LoCoMo"), ("beam-dev", "BEAM-1M")))
    invented = {m: round(sum(s["noise"][f"{m}@512"]["invented"] * s["noise"][f"{m}@512"]["n"] for s in answers.values())) for m in ("kanerva", "kanerva+gate", "hybrid")}
    foreign = sum(s["noise"]["kanerva@512"]["n"] for s in answers.values())
    versus_bm25 = final_answers["tests"]["kanerva vs bm25"]["contain"]["mean"]
    against_hybrid = ", ".join(f"{points(answers[name]['tests']['kanerva vs hybrid']['contain']['mean']['diff'])} on {title}" for name, title in ANSWER_SETS.items())
    text = f"""<img src="docs/assets/recall.png" alt="Evidence recovered at each token budget on the third held-out test and the BEAM check" width="100%">

Share of the needed evidence that made it into the context. These are retrieval metrics, with no LLM in the loop. The one exception is the answer test at the end of this list, where a small local model reads the notes.

- **Code history** is the third held-out test: {code['meta']['queries']:,} queries over 15 repositories in Python, TypeScript, Go and Rust that nothing was fitted or tuned on, shown without keys.
- **Long conversations** are {beam['meta']['queries']:,} questions over 55 BEAM conversations. They were never fitted on, but earlier tests saw them, so this is a check and not a test. No untouched conversation benchmark is left.

"Kanerva" is the default, with MiniLM. "Lexical" is what runs with `KANERVA_EMBEDDER=off` or until the model arrives. "Kanerva 0.4.1" is the previous release. The reranker was not rerun on BEAM.

{chr(10).join(rows)}

- **Code history.** Without keys, Kanerva leads hybrid search by {100 * plain['kanerva vs hybrid']['mean']['diff']:.1f} points over all budgets (interval {span(plain['kanerva vs hybrid']['mean'])}), ahead on all 15 repositories, and the reranker by {100 * plain['kanerva vs rerank']['mean']['diff']:.1f} points. With keys the lead over hybrid search is {100 * keyed['kanerva vs hybrid']['mean']['diff']:.1f} points. Against 0.4.1 it gains {100 * plain['kanerva vs kanerva-0.4']['mean']['diff']:.1f} points without keys and {100 * keyed['kanerva vs kanerva-0.4']['mean']['diff']:.1f} with them.
- **Each language.** Over all budgets, against hybrid search: {versus} Exact code facts are measured on Python only, because that track reads Python source.
- **C and C++.** A separate test on six repositories that nothing was tuned on, redis, curl, jq, fmt, spdlog and leveldb, with facts read by regular expressions. Without keys Kanerva leads hybrid search by {100 * c_plain['kanerva vs hybrid']['mean']['diff']:.1f} points over all budgets (interval {span(c_plain['kanerva vs hybrid']['mean'])}, all 6 repositories) and the reranker by {100 * c_plain['kanerva vs rerank']['mean']['diff']:.1f}, though it trails the reranker at 128 tokens. On tasks at 1024 tokens it has {fmt(c_plain['tracks']['task/kanerva@1024'])} against {fmt(c_plain['tracks']['task/hybrid@1024'])}. Lexical search only ties BM25, and it is weaker on C++, so let the model arrive there. Stale injection is {percent(c_plain['stale_on_changed_facts']['kanerva']['stale_any'])}% against {percent(c_plain['stale_on_changed_facts']['hybrid']['stale_any'])}% for hybrid search, and 0% with keys. File notes add {100 * c_files['with_files_vs_wider']['mean']:.1f} points at equal tokens. CMake, ctest, `make check` and Ninja runs count as test commands.
- **Long conversations.** {fmt(sum(beam['summary'][f'kanerva@{b}']['recall'] for b in BUDGETS) / len(BUDGETS))} against {fmt(sum(beam['summary'][f'hybrid@{b}']['recall'] for b in BUDGETS) / len(BUDGETS))} for hybrid search over all budgets, and {fmt(beam['summary']['kanerva@1024']['recall'])} against {fmt(beam['summary']['hybrid@1024']['recall'])} at 1024 tokens. The refit changed little here ({points(check['kanerva vs kanerva-0.4']['mean']['diff'])} points against 0.4.1). On LoCoMo, scored out of fold, it lost a little: {fmt(locomo[0])} to {fmt(locomo[1])}.
- **Stale facts.** At 1024 tokens on facts that changed, no outdated version got in with keys. Without keys, hybrid search injects one {percent(stale['hybrid']['stale_any'])}% of the time, Kanerva {percent(stale['kanerva']['stale_any'])}% and lexical {percent(stale['kanerva-lex']['stale_any'])}%. A one-line keep-the-newest rule gets {percent(stale['hybrid-dedup']['stale_any'])}%, so most of this comes from a simple idea. The store alone does not detect free-text updates, so both versions get injected, newest first, until your agent fixes the older one.
- **A correction.** This page used to give {percent(load('new_tests')['repomem-new-plain']['stale_on_changed_facts']['kanerva']['stale_any'])}% stale next to the lexical plugin default. That figure held only with MiniLM vectors. Benchmark stores held vectors even in "lexical" runs, so those rows compared near-twins by vectors, which a real lexical store does not have. The real lexical figure for 0.4.1 was {percent(stale['kanerva-lex-0.4']['stale_any'])}% on the third test. Lexical runs now compare near-twins by hashed words, as a real store does.
- **Prompts it cannot answer.** On 12 sampled streams of the third test, the gate keeps Kanerva silent on {percent(1 - full['with']['same']['spoke'])}% of questions from other repositories ({percent(1 - lexical['with']['same']['spoke'])}% lexical), where it used to speak on nearly all. This costs recall on a store's own questions: {fmt(full['without']['own']['recall'])} to {fmt(full['with']['own']['recall'])}, or {fmt(lexical['without']['own']['recall'])} to {fmt(lexical['with']['own']['recall'])} lexical. Scored out of fold on the earlier TypeScript, Go and Rust repositories the cost was larger, {fmt(spent['without']['own']['recall'])} to {fmt(spent['with']['own']['recall'])}. The false-alarm numbers come from foreign questions, not from real prompts that a store cannot answer.
- **Notes about files.** When a session has just opened a file of the change, file notes add {100 * file_now['with_files_vs_wider']['mean']:.1f} points over spending the same tokens on the prompt, on all 15 repositories. When it is still on the files of the task before, they cost {-100 * file_old['with_files_vs_wider']['mean']:.1f} points, again on all 15. Task labels come from file overlap, which is weak and favours this feature. Adding file names to the prompt query was tested and dropped, because it hurts in the second case.
- **Changed notes without keys.** `memory_remember` lists similar notes, and your agent retires or corrects the older one. In {agent['meta']['trials']} trials of Claude Code with Sonnet 5.5 on notes nothing was tuned on, it fixed {own['changed']['replaced'] + own['changed']['edited']} of {own['changed']['n']} changed facts in its own notes about a conversation, {verbatim['changed']['replaced'] + verbatim['changed']['edited']} of {verbatim['changed']['n']} in long word-for-word turns, where it edits out the outdated part, and {code_notes['changed']['replaced'] + code_notes['changed']['edited']} of {code_notes['changed']['n']} on code facts. In an earlier round it fixed none of {first['lme-updates']['changed']['n']} conversation notes, and the hint, the server's instructions and the list threshold were changed after that. In {touched} trials it also changed a note the labels did not mark. Reading them, most were renames, second copies of the old value or notes repeated by the new one, and one retired a signature along with a moved function, so that note survives only as history. The samples are small, and without vectors the list misses most paraphrases. An oracle that always marks the older version would take stale injection from {percent(changed['current_rule']['stale_any'])}% to {percent(changed['oracle_simulated']['stale_any'])}%. The list costs about {notes['list_tokens_when_shown']:.0f} tokens each time it appears.
- **Real prompts.** On {real['meta']['prompts']} real prompts of one developer, replayed through the hooks, the gate separates far less than on the benchmarks: {percent(real['vectors']['all']['gated_spoke'])}% of a project's own prompts get notes against {percent(real['vectors']['foreign']['all']['gated_spoke'])}% of prompts from other projects, and Russian prompts over English notes get them {percent(real['vectors']['ru']['gated_spoke'])}% of the time. The use signal behind `KANERVA_LEARN` fired on {percent(real['vectors']['signal']['fired'])}% of notes the agent never saw, so it stays off.
- **Where it loses.** Lexical search is close to level with BM25 at 512 tokens and above ({points(plain['kanerva-lex vs bm25']['512']['diff'])} points at 512, {points(plain['kanerva-lex vs bm25']['4096']['diff'])} at 4096, and the intervals include zero). Its lead is at 128 and 256 tokens. At 1024 tokens lexical search is {-100 * plain['kanerva-lex vs kanerva-lex-0.4']['1024']['diff']:.1f} points behind 0.4.1, the price of its stricter near-twin rule. Without keys Kanerva stays a little behind on exact facts ({fmt(tracks['fact/kanerva@1024'])} against {fmt(tracks['fact/hybrid@1024'])} at 1024 tokens). It is just behind hybrid search at 256 tokens in Go and Rust, and at 128 tokens in Rust. RepoMem queries come from templates, so they favour word search, and real prompts are vaguer.
- **Answers from a local model.** Qwen3-4B answered from the notes each method selected, with no judge in the scoring. Better recall turned into better answers only weakly. On {final_answers['queries']} fact queries from the third test, Kanerva leads BM25 by {100 * versus_bm25['diff']:.1f} points over 256, 512 and 1024 tokens (interval {span(versus_bm25)}, all 6 repositories), the only mean to reach raw p<.05, and no test survives Holm correction. On conversations it does not beat BM25, despite far higher recall: {conversations}, all inside their intervals. Against hybrid search it moves {against_hybrid}. An outdated value was given in {percent(final_stale['kanerva@512']['old'])}% of answers at 512 tokens against {percent(final_stale['hybrid@512']['old'])}% for hybrid search and {percent(final_stale['bm25@512']['old'])}% for BM25, on {final_stale['kanerva@512']['n']} changed facts, and a stale note in the context rarely became a stale answer. The gate saved tokens but did not stop made-up answers: {invented['kanerva']} of {foreign} foreign questions without it, {invented['kanerva+gate']} with it, {invented['hybrid']} for hybrid search. This reader is small and the samples are small, so read these as weak evidence, in either direction.

A query over 10,000 notes takes about {latency['hybrid@10000']['p50_ms']:.0f} ms with vectors or {latency['lexical@10000']['p50_ms']:.0f} ms lexical on an M1 Pro. A hook starts a new process each time, so a prompt with 10,000 notes takes {hooks['minilm@10000']['prompt']['p50_ms'] / 1000:.2f} s in all with MiniLM, {hooks['lexical@10000']['prompt']['p50_ms'] / 1000:.2f} s lexical and {hooks['static@10000']['prompt']['p50_ms'] / 1000:.2f} s with the static embedder."""
    path = ROOT / "README.md"
    readme = path.read_text()
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    head_text, rest = readme.split(start)
    _, tail = rest.split(end)
    path.write_text(f"{head_text}{start}\n{text}\n{end}{tail}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by scripts/make_macros.py; do not edit.",
        "\\newcommand{\\R}[3]{\\csname r-#1-#2-#3\\endcsname}",
        "\\newcommand{\\V}[1]{\\csname v-#1\\endcsname}",
    ]
    data = {n: load(f"fresh_{n}") for n in (*FRESH, *STATIC, *NEW, "repomem-new-static", *FINAL, "beam-check")}
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
    for name in (*FRESH, *STATIC):
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
    new_tests = load("new_tests")
    for name in (*NEW, "repomem-new-static"):
        for pair, entry in new_tests[name].items():
            if pair == "stale_on_changed_facts":
                for m, row in entry.items():
                    define(lines, f"v-nst-{name}-{m}", fmt(row["stale_any"]))
                continue
            key = pair.replace(" ", "_")
            for budget, t in entry.items():
                define(lines, f"v-nd-{name}-{key}-{budget}", signed(t["diff"]))
                define(lines, f"v-nci-{name}-{key}-{budget}", f"[{signed(t['cluster_ci'][0])}, {signed(t['cluster_ci'][1])}]")
                define(lines, f"v-npos-{name}-{key}-{budget}", t["units_positive"])
                define(lines, f"v-np-{name}-{key}-{budget}", pvalue(t.get("p_holm", t["p"])))
    for name, entry in load("new_tracks")["results"].items():
        for key, value in entry.items():
            track, rest = key.split("/")
            m, b = rest.split("@")
            define(lines, f"v-ntrack-{name}-{track}-{m}-{b}", fmt(value))
    final_tests = load("final_tests")
    for name in FINAL:
        for pair, entry in final_tests[name].items():
            if pair == "stale_on_changed_facts":
                for m, row in entry.items():
                    for k, v in row.items():
                        define(lines, f"v-fst-{name}-{m}-{k}", fmt(v))
                        define(lines, f"v-fstp-{name}-{m}-{k}", percent(v))
                continue
            key = pair.replace(" ", "_")
            for budget, t in entry.items():
                define(lines, f"v-fd-{name}-{key}-{budget}", signed(t["diff"]))
                define(lines, f"v-fci-{name}-{key}-{budget}", f"[{signed(t['cluster_ci'][0])}, {signed(t['cluster_ci'][1])}]")
                define(lines, f"v-fpos-{name}-{key}-{budget}", t["units_positive"])
                define(lines, f"v-fp-{name}-{key}-{budget}", pvalue(t.get("p_holm", t["p"])))
    for name, entry in load("final_tracks")["results"].items():
        for key, value in entry.items():
            track, rest = key.split("/")
            m, b = rest.split("@")
            define(lines, f"v-ftrack-{name}-{track}-{m}-{b}", fmt(value))
    languages = load("final_languages")["results"]
    for name, rows in languages.items():
        for language, entry in rows.items():
            for key, values in entry.items():
                if key == "queries":
                    continue
                track, m = key.split("/")
                for b, v in zip((*BUDGETS, "mean"), values):
                    define(lines, f"v-lang-{name}-{language}-{track}-{m}-{b}", fmt(v))
    (OUT / "table_final.tex").write_text(recall_table(data, FINAL, FINAL_SKIP))
    (OUT / "table_final_tests.tex").write_text(tests_table(final_tests, FINAL_PAIRS, ("128", "512", "1024", "4096", "mean"), FINAL))
    (OUT / "table_final_stale.tex").write_text(final_stale_table(final_tests))
    (OUT / "table_final_languages.tex").write_text(languages_table(languages["repomem-final-plain"]))
    for pair, rows in load("beam_check")["beam"].items():
        key = pair.replace(" ", "_")
        for budget, t in rows.items():
            define(lines, f"v-bcd-{key}-{budget}", signed(t["diff"]))
            define(lines, f"v-bcci-{key}-{budget}", f"[{signed(t['cluster_ci'][0])}, {signed(t['cluster_ci'][1])}]")
            define(lines, f"v-bcpos-{key}-{budget}", t["units_positive"])

    refit = load("refit")
    for kind in ("dense", "lexical"):
        for recipe, rows in refit[kind]["recipes"].items():
            for name, values in rows.items():
                define(lines, f"v-refit-{kind}-{recipe}-{name}", fmt(sum(values) / len(values)))

    gate_final, gate_spent = load("abstention_final"), load("abstention")
    gate_rows = {"final": {c: gate_final[c] for c in ("full", "lexical")}} | {name: entry for name, entry in gate_spent["sets"].items()}
    for name, configs in gate_rows.items():
        for config, entry in configs.items():
            for arm, kinds in entry.items():
                for kind, row in kinds.items():
                    for field, value in row.items():
                        text = fmt(value) if field in ("spoke", "recall") else f"{value:.0f}"
                        define(lines, f"v-gate-{name}-{config}-{arm}-{kind}-{field}", text)
                        if field == "spoke":
                            define(lines, f"v-gatep-{name}-{config}-{arm}-{kind}", percent(value))
            before, after = entry["without"]["own"]["recall"], entry["with"]["own"]["recall"]
            define(lines, f"v-gatecost-{name}-{config}", percent(1 - after / before))
    define(lines, "v-gate-keep", percent(1 - gate_spent["meta"]["keep"]))
    define(lines, "v-gate-budget", gate_spent["meta"]["budget"])
    (OUT / "table_gate.tex").write_text(gate_table(gate_spent, gate_final))

    file_sets = load("file_recall") | load("file_recall_final")
    define(lines, "v-file-budget", file_sets["meta"]["file_budget"])
    define(lines, "v-file-prompt-budget", file_sets["meta"]["prompt_budget"])
    for name, rows in file_sets.items():
        if name == "meta":
            continue
        for key, row in rows.items():
            session, config = key.split("/")
            prefix = f"v-file-{name}-{session}-{config}"
            for field in ("prompt", "wider", "with_files", "file_precision"):
                define(lines, f"{prefix}-{field}", fmt(row[field]))
            if config == "lexical":  # the embedding cache has no vectors for expanded queries
                define(lines, f"{prefix}-expanded", fmt(row["expanded"]))
            define(lines, f"{prefix}-queries", f"{row['queries']:,}".replace(",", "{,}"))
            define(lines, f"{prefix}-file_tokens", f"{row['file_tokens']:.0f}")
            define(lines, f"{prefix}-file_notes", f"{row['file_notes']:.1f}")
            move = row["with_files_vs_wider"]
            define(lines, f"{prefix}-gain", signed(move["mean"]))
            define(lines, f"{prefix}-up", f"{move['up']}/{move['repositories']}")
            define(lines, f"{prefix}-down", f"{move['down']}/{move['repositories']}")
    (OUT / "table_files.tex").write_text(files_table(file_sets))

    write_path = load("write_path")
    for arm in ("lexical", "vectors"):
        for name, entry in write_path[arm].items():
            similar = entry["similar"]
            prefix = f"v-wp-{arm}-{name}"
            for field in ("top1", "top3", "shown_on_other_notes", "top_hit_not_older", "top_hit_not_older_on_changed"):
                define(lines, f"{prefix}-{field}", fmt(similar[field]))
                define(lines, f"{prefix}-{field}-p", percent(similar[field]))
            define(lines, f"{prefix}-changed", f"{similar['changed_notes']:,}".replace(",", "{,}"))
            define(lines, f"{prefix}-tokens", f"{similar['list_tokens_when_shown']:.0f}")
            define(lines, f"{prefix}-ms", f"{similar['ms_per_call']:.0f}")
            for rule, row in entry["stale_on_changed_facts"].items():
                define(lines, f"{prefix}-{rule}-stale", fmt(row["stale_any"]))
                define(lines, f"{prefix}-{rule}-stale-p", percent(row["stale_any"]))
                define(lines, f"{prefix}-{rule}-queries", f"{row['queries']:,}".replace(",", "{,}"))
    (OUT / "table_write_path.tex").write_text(write_path_table(write_path))

    russian = load("russian")
    define(lines, "v-ru-articles", russian["xquad"]["articles"])
    define(lines, "v-ru-questions", f"{russian['xquad']['questions']:,}".replace(",", "{,}"))
    for language, row in russian["xquad"]["paragraphs"].items():
        define(lines, f"v-ru-tokens-{language}", f"{row['tokens_mean']:.0f}")
    for key, rows in russian["recall"].items():
        store, pair, method = key.split(" ")
        for budget, row in rows.items():
            define(lines, f"v-ru-{store}-{pair}-{method}-{budget}", fmt(row["recall"]))
    for key, metrics in russian["tests"].items():
        label = key.replace(":", "").replace(" ", "_")
        for budget, t in metrics["recall"].items():
            define(lines, f"v-rud-{label}-{budget}", signed(t["diff"]))
            define(lines, f"v-ruci-{label}-{budget}", f"[{signed(t['cluster_ci'][0])}, {signed(t['cluster_ci'][1])}]")
            define(lines, f"v-rupos-{label}-{budget}", t["units_positive"])
    for name, entry in russian["english_sample"].items():
        for budget, t in entry["multilingual_minus_minilm"].items():
            define(lines, f"v-rue-{name}-{budget}", signed(t["diff"]))
            define(lines, f"v-rueci-{name}-{budget}", f"[{signed(t['cluster_ci'][0])}, {signed(t['cluster_ci'][1])}]")
    for name, row in russian["size"].items():
        define(lines, f"v-ru-size-{name}", f"{row['total_mb']:.0f}")
    for name, row in russian["timing"].items():
        define(lines, f"v-ru-load-{name}", f"{row['load_s_median']:.2f}")

    answer_macros(lines, load("answers")["sets"])
    field_macros(lines)
    c_macros(lines)

    define(lines, "v-readme-stale", percent(new_tests["repomem-new-plain"]["stale_on_changed_facts"]["kanerva"]["stale_any"]))
    (OUT / "table_new.tex").write_text(recall_table(data, NEW, ("gen-agents", "facility", "recency", "mmr", "dense", "hybrid-dedup")))
    (OUT / "table_new_tests.tex").write_text(tests_table(new_tests, NEW_PAIRS, ("128", "1024", "4096", "mean"), NEW))
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
    calibration = load("dev_calibration")
    for name, row in calibration["results"].items():
        define(lines, f"v-ece-{name}", fmt(row["ece"], 4))
        define(lines, f"v-top-{name}-predicted", fmt(row["top_percent"]["predicted"]))
        define(lines, f"v-top-{name}-observed", fmt(row["top_percent"]["observed"]))
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
    rows = ["\\begin{tabular}{@{}lrrrrr@{}}\\toprule", "Dataset & " + " & ".join(ONLINE_COLUMNS.values()) + " \\\\\\midrule"]
    for d, settings in online["summary"].items():
        cells = []
        for s in ONLINE_COLUMNS:
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

    other = load("other_languages")
    other_recall = load("fresh_repomem-other")
    define(lines, "v-n-repomem-other", other_recall["meta"]["queries"])
    for key, row in other_recall["summary"].items():
        method, budget = key.split("@")
        define(lines, f"v-olr-{method}-{budget}", fmt(row["recall"]))
    for pair, rows in other["tests"].items():
        key = pair.replace(" ", "_")
        for budget, t in rows.items():
            define(lines, f"v-old-{key}-{budget}", signed(t["diff"]))
            define(lines, f"v-olci-{key}-{budget}", f"[{signed(t['cluster_ci'][0])}, {signed(t['cluster_ci'][1])}]")
            define(lines, f"v-olpos-{key}-{budget}", t["units_positive"])
    python_tasks = load("new_tracks")["results"]["repomem-new-plain"]
    define(lines, "v-ol-python-lead-1024", signed(python_tasks["task/kanerva@1024"] - python_tasks["task/hybrid@1024"]))

    twins = load("dev_static_twins")["table"]
    define(lines, "v-tw-settings", len(twins) - 1)
    for group, row in twins["vectors"].items():
        define(lines, f"v-tw-{group}-recall", fmt(row["recall"]))
        define(lines, f"v-tw-{group}-stale", fmt(row["stale_any"]))
    for group, row in twins["hashing-0.7-0.9"].items():
        define(lines, f"v-twh-{group}-recall", fmt(row["recall"]))
        define(lines, f"v-twh-{group}-stale", fmt(row["stale_any"]))
        define(lines, f"v-twh-{group}-diff", signed(row["test"]["diff"]))
        define(lines, f"v-twh-{group}-ci", f"[{signed(row['test']['cluster_ci'][0])}, {signed(row['test']['cluster_ci'][1])}]")

    latency = load("scale")
    for key, entry in latency["results"].items():
        define(lines, f"v-p50-{key}", f"{entry['p50_ms']:.1f}")
        define(lines, f"v-p95-{key}", f"{entry['p95_ms']:.1f}")
        define(lines, f"v-wr-{key}", f"{entry['write_ms']:.2f}")
    define(lines, "v-encode", f"{latency['query_encode_ms']:.0f}")
    for key, entry in load("hook_latency")["results"].items():
        define(lines, f"v-hook-{key}", f"{entry['prompt']['p50_ms'] / 1000:.2f}")

    (OUT / "macros.tex").write_text("\n".join(lines) + "\n")
    update_readme(data, latency["results"])


if __name__ == "__main__":
    sys.exit(main())
