<p align="center">
  <img src="docs/assets/banner.png" alt="Kanerva: long-term memory for coding agents" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-a8431d" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-b98a5e" alt="Python 3.10+">
  <a href="https://github.com/katolikov/kanerva/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/katolikov/kanerva/ci.yml?branch=main&label=tests&color=5f7f1a" alt="Tests"></a>
</p>

Coding agents start every session from zero. You explain the test command again, the deploy steps again, the bug you already fixed last week. Memory tools help, but most of them paste the top few similar notes into your context, whatever those notes cost and whether or not they are still true.

**Kanerva gives your agent long-term memory that runs on your machine, makes no LLM calls, and injects only what the current step needs, under a hard token budget.**

- **Small context.** You set the budget in tokens, and Kanerva picks the set of notes worth the most per token. Plugins estimate tokens instead of counting them exactly, so treat the limit as close, not exact.
- **Facts stay current.** Give a fact a key, or write a near-identical note, and the new version replaces the old one. Old versions stay in history, and every note shows its date, newest first.
- **Private by default.** One SQLite file in your home folder. No account, no cloud, no API key. Known key and token formats are redacted before anything is stored.
- **No extra model calls.** No LLM on write or read, so memory costs you only the tokens it injects.
- **Fits your tools.** Plugins for Claude Code and OpenCode, plus an MCP server for everything else.

## Install

You need [uv](https://docs.astral.sh/uv/) (recommended) or `python3` with NumPy.

### Claude Code

Inside Claude Code:

```text
/plugin install kanerva --marketplace katolikov/kanerva
```

Or from your shell:

```bash
claude plugin marketplace add katolikov/kanerva
claude plugin install kanerva@kanerva
```

Restart Claude Code. Memory now loads on session start and on each prompt, edits and test commands are captured as you work, and you get `/kanerva:memory`, `/kanerva:remember` and `/kanerva:forget`.

### OpenCode

```bash
curl -fsSL https://raw.githubusercontent.com/katolikov/kanerva/main/install.sh | sh -s -- opencode
```

Tested on OpenCode 1.19 and 2.2. Restart OpenCode and the eight `memory_*` tools are there, with no MCP setup. On V2 you can also install it as a package:

```bash
opencode plugin add 'github:katolikov/kanerva#main::path:plugins/opencode'
```

See the [plugin README](plugins/opencode/README.md) for details.

### Any MCP client

```bash
uv tool install git+https://github.com/katolikov/kanerva
```

```json
{
  "mcpServers": {
    "kanerva": { "command": "kanerva", "args": ["mcp"] }
  }
}
```

## What it looks like

<p align="center">
  <img src="docs/assets/terminal.png" alt="Kanerva in the terminal: a changed fact supersedes the old one and a token is redacted" width="82%">
</p>

There is no notes file to maintain, and everything stays readable and editable with `kanerva list`, `kanerva show <id>`, `kanerva edit <id> "..."`, `kanerva forget <id>` and `kanerva export memory.md`.

## How it works

<p align="center">
  <img src="docs/assets/how-it-works.png" alt="How Kanerva captures notes and selects context" width="100%">
</p>

Search engines find text that looks like your query. Kanerva also asks what is worth paying for. A small gradient-boosted model reads up to 29 signals for each candidate, such as how it ranked, how long it is, who said it, what sits next to it in the session and whether it mentions the entities in your question. The result is a probability that the note is needed. Selection then packs the most expected value into your token budget.

The model ships as a small JSON file and runs on NumPy, so there is no training step on your side and nothing heavy to install. Each project gets its own memory, and a `global` scope is shared across projects.

The name comes from Pentti Kanerva, whose sparse distributed memory showed how a few active bits out of many can recall what you need. It is also the idea behind the logo.

## Benchmarks

<!-- results:start -->
<img src="docs/assets/recall.png" alt="Evidence recovered at each token budget on held-out data" width="100%">

Share of the needed evidence that made it into the context, on held-out data: 984 questions over 55 BEAM conversations and 1,248 queries over five Python repositories, none of which were used for fitting. These are retrieval metrics, with no LLM in the loop, and code history is shown without keys. "Kanerva, full" uses the MiniLM sentence-embedding model through the library. The plugins run the two lighter setups.

| Dataset | Budget | Kanerva, full | Plugin default | Plugin, static embedder | Hybrid + reranker | Hybrid | BM25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Long conversations | 256 | **0.449** | 0.388 | 0.438 | 0.439 | 0.428 | 0.371 |
| Long conversations | 1024 | **0.635** | 0.546 | 0.592 | 0.488 | 0.430 | 0.368 |
| Long conversations | 4096 | **0.754** | 0.641 | 0.705 | 0.576 | 0.529 | 0.431 |
| Code history | 256 | 0.786 | 0.758 | 0.715 | 0.796 | **0.799** | 0.763 |
| Code history | 1024 | 0.855 | 0.833 | 0.804 | **0.871** | 0.869 | 0.850 |
| Code history | 4096 | 0.917 | 0.895 | 0.890 | **0.951** | **0.951** | 0.939 |

- **Long conversations are where it shines.** From 1024 tokens up, Kanerva pulls well ahead of everything else, with 0.635 against 0.488 for hybrid search with a cross-encoder reranker at 1024 tokens. Below that it is level with the reranker, without needing one. Over all budgets it recovers 9 points more.
- **Stale facts mostly stop, when they can be detected.** At 1024 tokens on facts that changed, hybrid search injects an outdated version 99.7% of the time. Kanerva does so 19.4% of the time without keys, and on none of the test queries with keys. A one-line keep-the-newest rule gets 22.7%, so most of this comes from a simple idea. Free-text updates are not detected: both versions get injected, newest first.
- **Where it loses.** On code history, the rule that removes near-identical notes also drops useful look-alike commits, so large budgets favour plain search: 0.917 against 0.951 at 4096 tokens. The plugin default is about level with BM25 there, and behind it at large budgets (0.895 against 0.939). The static embedder helps on conversations but hurts on code (0.804 against 0.833 at 1024 tokens) and lets through more stale facts (37.1%). With five repositories the intervals are wide, so read differences of a few points on code as ties.

A query over 10,000 notes takes about 150 ms on an M1 Pro, or 287 ms with embeddings. A hook call adds Python start-up on top, for 0.2 to 0.4 s in total.
<!-- results:end -->

The full method, proofs and all results are in the [paper](paper/paper.pdf).

## Configuration

Everything is optional.

| Variable | Default | What it does |
|---|---|---|
| `KANERVA_EMBEDDER` | off | Set to `static` to add vector search with [model2vec](https://github.com/MinishLab/model2vec). Good for long conversational notes, worse for exact code facts (see [Benchmarks](#benchmarks)). Downloads a small model on first use. |
| `KANERVA_PROMPT_BUDGET` | `600` | Tokens injected per prompt. |
| `KANERVA_SESSION_BUDGET` | `400` | Tokens injected at session start. |
| `KANERVA_MIN_WEIGHT` | `0.02` | Skip notes below this estimated probability of being needed. |
| `KANERVA_HOME` | `~/.kanerva` | Where the database lives. |
| `KANERVA_CONSOLIDATE` | off | Set to `1` to merge near-duplicate notes on compaction. Merges can be undone. |
| `KANERVA_LEARN` | off | Set to `1` to learn from which notes were used. |

## Use it as a library

```python
from kanerva import Memory

memory = Memory("memory.sqlite")
memory.write("The API timeout is 30 seconds", key="config:timeout")
memory.write("The API timeout is 60 seconds", key="config:timeout")  # supersedes the old value, keeps history
print(memory.retrieve("what is the API timeout?", budget=200).render())
```

For the best recall at small budgets, pass `reranker=CrossEncoderReranker()` with the `transformers` extra. It costs one local model call per query.

## Compared to other memory tools

| | Kanerva | Claude Code files | Mem0 | Zep / Graphiti | Letta |
|---|---|---|---|---|---|
| LLM needed on write | no | model writes the notes | yes, for fact extraction | yes | agent tool calls |
| Hard token budget | yes | 200 lines or 25 KB, unranked | no, top-k | no, top-k | block sizes |
| Handles changed facts | version chains, near-twin rule | no | LLM update or delete | bi-temporal edges | agent-managed |
| Runs without a service | yes | yes | yes, with a local vector store | needs a graph DB | server or CLI |

Those projects have things Kanerva does not, such as LLM extraction, graph traversal and hosted operation.

## Reproduce the results

```bash
make install && make data
make reproduce
make tools && make paper && make check
```

`make check` confirms that every number in the paper comes from `results/`.

## Contributing

Issues and pull requests are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md). Benchmark results on your own repositories are especially useful.

## Citation

```bibtex
@misc{kanerva2026,
  title  = {Kanerva: Token-Budgeted, Validity-Aware Memory for Coding Agents, Without an LLM in the Loop},
  author = {Katolikov, Artem},
  year   = {2026},
  note   = {Preprint}
}
```

## License

[Apache-2.0](LICENSE). The LoCoMo and BEAM benchmarks have their own licenses and are downloaded when you run the experiments, not shipped here.
