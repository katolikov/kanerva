"""Recall-against-budget figures for the paper and the README."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BUDGETS = (128, 256, 512, 1024, 2048, 4096)
PANELS = {"fresh_beam": "Long conversations (BEAM, 984 questions)", "fresh_repomem-fresh-plain": "Code history (RepoMem, 1,248 questions)"}
SERIES = (
    ("kanerva-ce", "Kanerva + reranker", 0),
    ("kanerva", "Kanerva", 1),
    ("rerank", "Hybrid + reranker", 2),
    ("hybrid", "Hybrid", 3),
    ("bm25", "BM25", 4),
)
THEMES = {
    "light": {"surface": "#ffffff", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e0",
              "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")},
    "dark": {"surface": "#0d1117", "ink": "#f0efec", "muted": "#c3c2b7", "grid": "#2a2f38",
             "series": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181")},
}


def draw(path: Path, theme: dict, width: float = 9.6, height: float = 4.1) -> None:
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 11})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(width, height), sharey=True, facecolor=theme["surface"])
    handles = []
    for ax, (name, title) in zip(axes, PANELS.items()):
        summary = json.loads((RESULTS / f"{name}.json").read_text())["summary"]
        ax.set_facecolor(theme["surface"])
        for method, label, slot in SERIES:
            if f"{method}@128" not in summary:
                continue
            values = [summary[f"{method}@{b}"]["recall"] for b in BUDGETS]
            line, = ax.plot(BUDGETS, values, color=theme["series"][slot], linewidth=2, marker="o", markersize=7.5,
                            markeredgecolor=theme["surface"], markeredgewidth=1.5, label=label, zorder=3 + (5 - slot))
            if label not in [h.get_label() for h in handles]:
                handles.append(line)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        ax.minorticks_off()
        ax.set_ylim(0.2, 1.0)
        ax.set_title(title, color=theme["ink"], fontsize=12, fontweight="bold", loc="left", pad=12)
        ax.set_xlabel("Token budget", color=theme["muted"])
        ax.grid(axis="y", color=theme["grid"], linewidth=1)
        ax.tick_params(colors=theme["muted"], length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(theme["grid"])
    axes[0].set_ylabel("Evidence recovered", color=theme["muted"])
    legend = fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.01))
    for text in legend.get_texts():
        text.set_color(theme["ink"])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=theme["surface"])
    plt.close(fig)


def main() -> None:
    for name, theme in THEMES.items():
        draw(ROOT / "docs" / "assets" / f"recall-{name}.png", theme)
    draw(ROOT / "paper" / "figs" / "recall.pdf", THEMES["light"], width=7.2, height=3.3)


if __name__ == "__main__":
    main()
