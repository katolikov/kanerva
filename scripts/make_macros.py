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
             "repomem-fresh": "RepoMem, earlier test 1", "repomem-new": "RepoMem, earlier test 2", "repomem-other": "RepoMem beyond Python"}
FILE_SETS = {"repomem-final": "Main test", "repomem-other": "Beyond Python", "repomem-new": "Earlier test 2", "repomem-fresh": "Earlier test 1",
             "repomem-holdout": "Holdout", "repomem": "RepoMem dev"}
WRITE_SETS = {"repomem-final": "RepoMem, main test", "repomem-new": "RepoMem, earlier test 2", "repomem-fresh": "RepoMem, earlier test 1", "lme-updates": "LongMemEval updates"}
ONLINE_COLUMNS = {"static": "Static", "online": "Bandit", "online-noisy": "Bandit, noisy", "lift": "Lift", "lift-noisy": "Lift, noisy"}
ANSWER_SETS = {"locomo": "LoCoMo", "lme": "LongMemEval-S", "beam-dev": "BEAM-1M", "repomem-final": "RepoMem, main test"}
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
        entries = [("Main test", final[config])] + [(label, spent["sets"][name][config]) for name, label in GATE_SETS.items()]
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


C_METHODS = {"kanerva": "\\kanerva", "kanerva-ce": "\\kanerva + reranker", "kanerva-lex": "\\kanerva, lexical", "kanerva-0.5": "\\kanerva 0.5.0", "kanerva-lex-0.5": "\\kanerva 0.5.0, lexical", "kanerva-0.4": "\\kanerva 0.4.1", "rerank": "Hybrid + reranker",
             "hybrid": "Hybrid RRF", "hybrid-dedup": "Hybrid, newest of twins", "bm25": "BM25"}
C_PAIRS = ("kanerva vs hybrid", "kanerva vs rerank", "kanerva vs kanerva-0.4", "kanerva-lex vs bm25", "kanerva-lex vs kanerva-lex-0.4",
           "kanerva vs kanerva-0.5", "kanerva-lex vs kanerva-lex-0.5")


def c_table(summary: dict) -> str:
    rows = ["\\begin{tabular}{@{}l" + "r" * (len(BUDGETS) + 1) + "@{}}\\toprule", "Method & " + " & ".join(map(str, BUDGETS)) + " & mean \\\\\\midrule"]
    methods = [m for m in C_METHODS if f"{m}@128" in summary]
    best = {b: max(summary[f"{m}@{b}"]["recall"] for m in methods) for b in BUDGETS}
    best_mean = max(mean_over_budgets(summary, m) for m in methods)
    for m in methods:
        cells = [f"\\textbf{{{fmt(summary[f'{m}@{b}']['recall'])}}}" if abs(summary[f"{m}@{b}"]["recall"] - best[b]) < 5e-4 else fmt(summary[f"{m}@{b}"]["recall"]) for b in BUDGETS]
        mean = mean_over_budgets(summary, m)
        cells.append(f"\\textbf{{{fmt(mean)}}}" if abs(mean - best_mean) < 5e-4 else fmt(mean))
        rows.append(f"{C_METHODS[m]} & " + " & ".join(cells) + " \\\\")
    rows.append("\\bottomrule\\end{tabular}")
    return "\n".join(rows)


def c_macros(lines: list[str]) -> None:
    """C and C++: both tests, the dev group, file notes, the gate, and the check of the near-twin rule on older data."""
    reports = {"test": load("c-test_tests"), "dev": load("c-dev_tests"), "final": load("c-final_tests"), "last": load("c-last_tests")}
    data = {"test": load("fresh_repomem-c-test-plain"), "final": load("fresh_repomem-c-final-plain"), "last": load("fresh_repomem-c-last-plain")}
    for name, entry in data.items():
        define(lines, f"v-c-queries-{name}", f"{entry['meta']['queries']:,}".replace(",", "{,}"))
        for key, row in entry["summary"].items():
            method, budget = key.split("@")
            define(lines, f"v-c-r-{name}-{method}-{budget}", fmt(row["recall"]))
        for method in sorted({k.split("@")[0] for k in entry["summary"]}):
            define(lines, f"v-c-r-{name}-{method}-mean", fmt(mean_over_budgets(entry["summary"], method)))
    for name, report in reports.items():
        for label, entry in (("plain", report[f"repomem-c-{name}-plain"]), ("keys", report[f"repomem-c-{name}"])):
            for pair in C_PAIRS:
                if pair not in entry:
                    continue
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
            for method, value in entry.get("answers", {}).items():
                define(lines, f"v-cans-{name}-{label}-{method}", fmt(value, 4))
        for language, entry in report.get("languages", {}).items():
            safe = language.replace("+", "p")
            for key, values in entry.items():
                if key != "queries":
                    track, method = key.split("/")
                    define(lines, f"v-clang-{name}-{safe}-{track}-{method}", fmt(values[-1]))
    recall_files = load("file_recall_c")
    for name in ("test", "final", "last"):
        files = recall_files[f"repomem-c-{name}"]
        for session in ("one", "previous"):
            for config in ("full", "lexical"):
                row = files[f"{session}/{config}"]
                prefix = f"v-cf-{name}-{session}-{config}"
                for field in ("prompt", "wider", "with_files"):
                    define(lines, f"{prefix}-{field}", fmt(row[field]))
                move = row["with_files_vs_wider"]
                define(lines, f"{prefix}-gain", signed(move["mean"]))
                define(lines, f"{prefix}-up", f"{move['up']}/{move['repositories']}")
                define(lines, f"{prefix}-down", f"{move['down']}/{move['repositories']}")
    for name, path in (("test", "abstention_c"), ("final", "abstention_c_final"), ("last", "abstention_c_last")):
        gate = load(path)
        for config in ("full", "lexical"):
            for arm in ("without", "with"):
                for kind in ("own", "same", "other"):
                    define(lines, f"v-cg-{name}-{config}-{arm}-{kind}", fmt(gate[config][arm][kind]["spoke"]))
                define(lines, f"v-cg-{name}-{config}-{arm}-recall", fmt(gate[config][arm]["own"]["recall"]))
    refit = load("refit_c")
    for kind in ("dense", "lexical"):
        old, new = refit[kind]["recipes"]["all-families"], refit[kind]["recipes"]["with-c"]
        for name in old:
            define(lines, f"v-rc-{kind}-{name}", signed((sum(new[name]) - sum(old[name])) / len(old[name])))
    twins = load("twin_rule")
    for label, key in (("third test", "third"), ("c-test", "ctest"), ("beam", "beam")):
        entry = twins[label]
        define(lines, f"v-tr-{key}-dense", signed(entry["kanerva vs kanerva-0.5"]["diff"]))
        define(lines, f"v-tr-{key}-lex", signed(entry["kanerva-lex vs kanerva-lex-0.5"]["diff"]))
        for method, value in entry["stale"].items():
            define(lines, f"v-tr-{key}-stale-{method}", percent(value))
        for method, value in entry["answers"].items():
            define(lines, f"v-tr-{key}-ans-{method}", fmt(value, 4))
    agent = load("agent_write_path_c")
    define(lines, "v-awpc-trials", agent["meta"]["trials"])
    entry = agent["sets"]["repomem-c-final"]
    define(lines, "v-awpc-fixed", entry["changed"]["replaced"] + entry["changed"]["edited"])
    define(lines, "v-awpc-n", entry["changed"]["n"])
    define(lines, "v-awpc-other-n", entry["other"]["n"])
    define(lines, "v-awpc-other-clean", entry["other"]["n"] - entry["other"]["extra"])
    (OUT / "table_c.tex").write_text(c_table(data["test"]["summary"]))
    (OUT / "table_c_final.tex").write_text(c_table(data["final"]["summary"]))
    (OUT / "table_c_last.tex").write_text(c_table(data["last"]["summary"]))
    losses = load("fact_losses")
    define(lines, "v-cl-facts", f"{losses['facts']:,}".replace(",", "{,}"))
    define(lines, "v-cl-lost", losses["lost"])
    define(lines, "v-cl-locations", losses["lost_locations"])
    define(lines, "v-cl-value", fmt(losses["value_in_context"]))


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
    plain = load("final_tests")["repomem-final-plain"]
    hooks = load("hook_latency")["results"]
    final_c, last_c = load("c-final_tests"), load("c-last_tests")
    c_lang, c_earlier = last_c["languages"], final_c["repomem-c-final-plain"]["kanerva vs hybrid"]
    c_stale = last_c["repomem-c-last-plain"]["stale_on_changed_facts"]
    c_files = load("file_recall_c")["repomem-c-last"]["one/full"]
    c_agent, c_agent_trials = load("agent_write_path_c")["sets"]["repomem-c-final"], load("agent_write_path_c")["meta"]["trials"]
    twins = load("twin_rule")
    agent = load("agent_write_path")
    own, verbatim, code_notes = agent["sets"]["lme-own-notes"], agent["sets"]["lme-updates"], agent["sets"]["repomem-final"]
    real = load("real_sessions")
    signal = load("agent_signal")
    chain_table = load("agent_chain")["budgets"]["default"]
    plain_questions = load("plain_questions")["settings"]
    plain_pool = load("plain_questions")["in_pool"]
    text = f"""<img src="docs/assets/recall.png" alt="Evidence recovered at each token budget on the main test and the BEAM check" width="100%">

Share of the needed evidence that made it into the context, with no LLM in the loop. "Lexical" is what runs until the model arrives. Unless noted, these figures come from version 0.5.0.

{chr(10).join(rows)}

- **Code.** On {code['meta']['queries']:,} queries over 15 repositories in Python, TypeScript, Go and Rust that nothing was tuned on, Kanerva leads hybrid search by {100 * plain['kanerva vs hybrid']['mean']['diff']:.1f} points over all budgets, on all 15, and the reranker by {100 * plain['kanerva vs rerank']['mean']['diff']:.1f}.
- **C and C++.** On six more repositories, run once on the current version, Kanerva leads clearly on C, {fmt(c_lang['C']['all/kanerva'][-1])} against {fmt(c_lang['C']['all/hybrid'][-1])} for hybrid search. On C++, with templates and class methods among the facts, it leads on tasks ({fmt(c_lang['C++']['task/kanerva'][-1])} against {fmt(c_lang['C++']['task/hybrid'][-1])}) but trails overall ({fmt(c_lang['C++']['all/kanerva'][-1])} against {fmt(c_lang['C++']['all/hybrid'][-1])}), mostly because a location note gives way to a newer note holding the same value, so the value still reaches the context. An earlier test of six other repositories gave a lead of {100 * c_earlier['mean']['diff']:.1f} points on all 6. Notes about files add {100 * c_files['with_files_vs_wider']['mean']:.1f} points at equal tokens. Working CMake, make and Ninja commands are kept as notes.
- **Long conversations.** {fmt(sum(beam['summary'][f'kanerva@{b}']['recall'] for b in BUDGETS) / len(BUDGETS))} against {fmt(sum(beam['summary'][f'hybrid@{b}']['recall'] for b in BUDGETS) / len(BUDGETS))} for hybrid search. This is a check and not a test, because earlier tests saw the data.
- **Stale facts.** With keys, no outdated version gets in. Without keys at 1024 tokens, the current version injects one {percent(c_stale['kanerva']['stale_any'])}% of the time on C and C++ and {percent(twins['third test']['stale']['kanerva'])}% on the earlier test's repositories, against about {percent(c_stale['hybrid']['stale_any'])}% for hybrid search.
- **Changed notes.** `memory_remember` lists similar notes and your agent fixes the older one. Across {agent['meta']['trials'] + c_agent_trials} real Claude Code sessions, it fixed {own['changed']['replaced'] + own['changed']['edited']} of {own['changed']['n']} facts in its own conversation notes, {verbatim['changed']['replaced'] + verbatim['changed']['edited']} of {verbatim['changed']['n']} in long turns and {code_notes['changed']['replaced'] + code_notes['changed']['edited'] + c_agent['changed']['replaced'] + c_agent['changed']['edited']} of {code_notes['changed']['n'] + c_agent['changed']['n']} in code. It sometimes retires more than it should, and one such note survived only as history.
- **Learning.** Over {signal['trials']} real sessions on repositories nothing had seen, your agent's verdicts on notes were right {100 * signal['verdict']['precision']:.1f}% of the time, against {100 * signal['old_signal']['precision']:.1f}% for matching identifiers in tool calls. Acting on those verdicts did not pass its test: on six more repositories, sessions a note helped rose from {100 * chain_table['helped_sessions']['fixed']:.1f}% to {100 * chain_table['helped_sessions']['learned']:.1f}%, but right answers fell from {100 * chain_table['right']['fixed']:.1f}% to {100 * chain_table['right']['learned']:.1f}%. So by default Kanerva only records verdicts.
- **Where it falls short.** Asked in plain words with no identifier, the shipped gate brings the needed note on only {plain_questions['0.0/0.02']['found']} of {plain_questions['0.0/0.02']['n']} questions, though it reaches the candidate list on {plain_pool} of them. Lexical search stays close to BM25 from 512 tokens up. Without keys it gives up a little on exact facts. On answers from a small local model, better recall barely helped and no difference survived correction. On {real['meta']['prompts']} real prompts of one developer, the gate gave notes to {percent(real['vectors']['all']['gated_spoke'])}% of a project's own prompts and {percent(real['vectors']['foreign']['all']['gated_spoke'])}% of other projects' prompts, far weaker than on benchmarks. RepoMem queries come from templates, and real prompts are vaguer.

A query over 10,000 notes takes about {latency['hybrid@10000']['p50_ms']:.0f} ms, or {hooks['minilm@10000']['prompt']['p50_ms'] / 1000:.2f} s for a whole hook, which starts a new process."""
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

    verdicts = load("agent_signal")
    judged, old = verdicts["verdict"], verdicts["old_signal"]
    define(lines, "v-sig-trials", verdicts["trials"])
    define(lines, "v-sig-repos", len(verdicts["meta"]["repos"]))
    define(lines, "v-sig-coverage", f"{100 * verdicts['coverage']:.0f}")
    define(lines, "v-sig-other-n", judged["false_mark"]["n"])
    define(lines, "v-sig-unseen", verdicts["unseen_marks"])
    for label, entry in (("sig", judged), ("sigold", old)):
        define(lines, f"v-{label}-precision", f"{100 * entry['precision']:.1f}")
        define(lines, f"v-{label}-precision-ci", f"{100 * entry['precision_interval'][0]:.0f}--{100 * entry['precision_interval'][1]:.0f}")
        define(lines, f"v-{label}-recall", f"{100 * entry['recall']['share']:.1f}")
        define(lines, f"v-{label}-false", f"{100 * entry['false_mark']['share']:.1f}")
        define(lines, f"v-{label}-false-hits", entry["false_mark"]["hits"])
        define(lines, f"v-{label}-stale-helped", entry["stale_marked_helped"]["hits"])
    define(lines, "v-sig-stale-n", judged["stale_marked_misled"]["n"])
    define(lines, "v-sig-stale-misled", judged["stale_marked_misled"]["hits"])
    chain = load("agent_chain")
    gate_replay = load("real_gate")
    define(lines, "v-rs-recurrence", f"{100 * gate_replay['recurrence']['share']:.0f}")
    define(lines, "v-chain-recurrence", f"{100 * chain['meta']['recurrence']:.0f}")
    define(lines, "v-chain-sessions", chain["meta"]["sessions"])
    define(lines, "v-chain-repos", len(chain["meta"]["repos"]))
    define(lines, "v-chain-notes", chain["meta"]["store_notes"])
    table = chain["budgets"]["default"]
    units = len(chain["meta"]["repos"])
    scored = chain["meta"]["sessions"] // 2 * units
    define(lines, "v-chain-scored", scored)
    for measure, entry in table.items():
        if isinstance(entry, dict):
            digits = 1 if measure == "tokens" else 2 if measure in ("lookups", "shown") else 3
            for field in ("fixed", "learned", "delta"):
                define(lines, f"v-chain-{measure}-{field}", f"{entry[field]:+.{digits}f}" if field == "delta" else f"{entry[field]:.{digits}f}")
            define(lines, f"v-chain-{measure}-ci", f"[{entry['interval'][0]:+.{digits}f}, {entry['interval'][1]:+.{digits}f}]")
            define(lines, f"v-chain-{measure}-up", entry["up"])
            define(lines, f"v-chain-{measure}-down", entry["down"])
            define(lines, f"v-chain-{measure}-p", f"{entry['p']:.2f}")
    for field in ("fixed", "learned"):
        define(lines, f"v-chain-count-helped-{field}", round(table["helped_sessions"][field] * scored))
        define(lines, f"v-chain-count-right-{field}", round(table["right"][field] * scored))
    define(lines, "v-chain-lift-chains", chain["lift"]["chains"])
    define(lines, "v-chain-lift-sessions", chain["lift"]["scored_sessions"])
    first = load("agent_chain_first")["budgets"]["default"]
    first_scored = 72
    for field in ("fixed", "learned"):
        define(lines, f"v-chainfirst-helped-{field}", round(first["helped_sessions"][field] * first_scored))
        define(lines, f"v-chainfirst-right-{field}", round(first["right"][field] * first_scored))
    define(lines, "v-chainfirst-tokens", f"{first['tokens']['delta']:.0f}")
    plain_report = load("plain_questions")
    plain = plain_report["settings"]
    for label, setting in (("default", "0.0/0.02"), ("gate", "-3.0/0.02"), ("wide", "-3.0/0.002")):
        define(lines, f"v-pq-{label}-found", plain[setting]["found"])
        define(lines, f"v-pq-{label}-tokens", f"{plain[setting]['tokens']:.0f}")
    define(lines, "v-pq-n", plain["0.0/0.02"]["n"])
    define(lines, "v-pq-pool", plain_report["in_pool"])
    hazard = load("hazard_check")
    define(lines, "v-hz-vectors", f"{hazard['vectors']['diff']:+.3f}")
    define(lines, "v-hz-vectors-ci", f"[{hazard['vectors']['ci'][0]:+.3f}, {hazard['vectors']['ci'][1]:+.3f}]")
    define(lines, "v-hz-lexical", f"{hazard['lexical']['diff']:+.3f}")
    for method, label in (("kanerva", "with"), ("kanerva-no-hazard", "without")):
        define(lines, f"v-hz-stale-{label}", f"{100 * hazard['stale'][method]:.1f}")
    for arm in ("fixed", "learned"):
        for field, value in gate_replay[arm]["all"].items():
            if value is not None:
                define(lines, f"v-rg-{arm}-{field}".replace("_", "-"), f"{100 * value:.1f}" if field.startswith("spoke") else fmt(value, 1) if field == "tokens" else value)

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
