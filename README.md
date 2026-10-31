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
- **No extra model calls.** No LLM on write or read, so memory costs you the tokens it injects, plus one short agent turn when a session with real work ends. You can [turn that off](#session-wrap-up).
- **Fits your tools.** Plugins for Claude Code, Codex, OpenCode, Copilot CLI and more, plus an MCP server for everything else. The [table](#install) says which ones have been run.

## Install

You need [uv](https://docs.astral.sh/uv/) (recommended) or `python3` with NumPy. Then run this with the name of your agent, or with no name to set up every supported agent it finds:

```bash
curl -fsSL https://raw.githubusercontent.com/katolikov/kanerva/main/install.sh | sh -s -- codex
```

The script runs each agent's own install commands. Agents that need local files use a clone in `~/.kanerva/src`, and run it again to update.

| Agent | Name for the script | Notes injected | `memory_*` tools | How it was checked |
|---|---|---|---:|---|
| Claude Code | `claude` | every prompt | yes | run on 2.1.314 |
| OpenCode | `opencode` | every prompt | yes | run on 1.19 and 2.2 |
| Codex | `codex` | every prompt, once you trust the hooks | yes | run on 0.166 |
| OpenClaw | `openclaw` | every prompt | yes | run on 2026.10.9 |
| Copilot CLI | `copilot` | every prompt | yes | run on 1.0.105 |
| Cursor | `cursor` | session start, then at the first tool result of a prompt | yes | docs only |
| Hermes | `hermes` | every prompt | yes, see its README | run on 0.19.0 |
| Pi | `pi` | every prompt | yes | run on 1.4.0 |
| Qwen Code | `qwen` | every prompt | yes | run on 0.26.4 |
| Devin | `devin` | every prompt | yes | docs only |
| Gemini CLI | `gemini` | every prompt | yes | run on 0.66.0 |
| Antigravity | `antigravity` | session start only | yes | docs only |
| Amp | `amp` | every prompt | no | docs, plugin load only |
| Kiro CLI | `kiro` | every prompt | yes | docs only |
| Cline CLI | `cline` | start of each run | yes | run on 3.0.74 |
| Goose, Zed | [MCP setup](plugins/mcp/README.md) | no, the agent has to ask | yes | docs only |

"Run" means a real agent session against a local stub model, with a seeded note showing up in the model's request. "Docs only" means the setup follows the agent's documentation but has not been run, so expect rough edges and please report them. The agent's README under [`plugins/`](plugins) has its details and limits.

If you would rather use the agent's own commands:

```bash
# Claude Code
claude plugin marketplace add katolikov/kanerva
claude plugin install kanerva@kanerva

# Codex: start Codex afterwards and run /hooks, because hooks stay off until you trust them
codex plugin marketplace add katolikov/kanerva
codex plugin add kanerva@kanerva

# Copilot CLI
copilot plugin marketplace add katolikov/kanerva
copilot plugin install kanerva@kanerva

# Hermes
hermes plugins install katolikov/kanerva/plugins/hermes --enable

# OpenCode 2.x
opencode plugin add 'github:katolikov/kanerva#main::path:plugins/opencode'
```

For any other MCP client:

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

Share of the needed evidence that made it into the context, on held-out data: 984 questions over 55 BEAM conversations and 2,014 queries over eight Python repositories, none of which were used for fitting. These are retrieval metrics, with no LLM in the loop, and code history is shown without keys. "Kanerva, full" uses the MiniLM sentence-embedding model, which the plugins offer as a [setting](#minilm-in-the-plugins). They run the lexical default out of the box.

| Dataset | Budget | Kanerva, full | Plugin default | Plugin, static embedder | Hybrid + reranker | Hybrid | BM25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Long conversations | 256 | **0.449** | 0.388 | 0.438 | 0.439 | 0.428 | 0.371 |
| Long conversations | 1024 | **0.635** | 0.546 | 0.592 | 0.488 | 0.430 | 0.368 |
| Long conversations | 4096 | **0.754** | 0.641 | 0.705 | 0.576 | 0.529 | 0.431 |
| Code history | 256 | 0.800 | 0.778 | 0.751 | 0.813 | **0.815** | 0.783 |
| Code history | 1024 | 0.872 | 0.858 | 0.831 | 0.883 | **0.887** | 0.871 |
| Code history | 4096 | **0.935** | 0.908 | 0.903 | **0.935** | **0.935** | 0.917 |

- **Long conversations are where it shines.** From 1024 tokens up, Kanerva pulls well ahead of everything else, with 0.635 against 0.488 for hybrid search with a cross-encoder reranker at 1024 tokens. Below that it is level with the reranker, without needing one. Over all budgets it recovers 9 points more.
- **On code, keys make the difference.** With keys, Kanerva beats hybrid search on all eight repositories, by 2.1 points over all budgets. Without keys it is level with hybrid search and the reranker: ahead at 128 tokens, a little behind in the middle, and the intervals include zero.
- **Stale facts mostly stop, when they can be detected.** At 1024 tokens on facts that changed, hybrid search injects an outdated version 99.1% of the time. Kanerva does so 24.3% of the time without keys, and on none of the test queries with keys. A one-line keep-the-newest rule gets 31.9%, so most of this comes from a simple idea. Free-text updates are not detected: both versions get injected, newest first.
- **Where it loses.** The near-twin rule still costs a little on exact code facts without keys (0.975 against 0.997 at 4096 tokens). The plugin default is level with BM25 on code and slightly behind it at 1024 tokens and above (0.858 against 0.871). The static embedder helps on conversations, hurts on code (0.831 against 0.858 at 1024 tokens) and lets through more stale facts (37.4%). Comparing near-twins by hashed words instead raises recall but does not reliably lower that (49.5% on these repositories at the closest setting), so it is not shipped. All repositories are Python.

A query over 10,000 notes takes about 150 ms on an M1 Pro, or 287 ms with embeddings. A hook starts a new process each time, so a prompt with 10,000 notes takes 0.45 s in all on the default setup, 0.67 s with MiniLM and 0.92 s with the static embedder.
<!-- results:end -->

The full method, proofs and all results are in the [paper](paper/paper.pdf).

## Configuration

Everything is optional.

| Variable | Default | What it does |
|---|---|---|
| `KANERVA_EMBEDDER` | off | Set to `minilm` to add vector search with the [MiniLM model](#minilm-in-the-plugins), or to `static` for [model2vec](https://github.com/MinishLab/model2vec), which is good for long conversational notes and worse for exact code facts (see [Benchmarks](#benchmarks)). Either one downloads a model on first use. |
| `KANERVA_PROMPT_BUDGET` | `600` | Tokens injected per prompt. |
| `KANERVA_SESSION_BUDGET` | `400` | Tokens injected at session start. |
| `KANERVA_MIN_WEIGHT` | `0.02` | Skip notes below this estimated probability of being needed. |
| `KANERVA_HOME` | `~/.kanerva` | Where the database lives. |
| `KANERVA_CONSOLIDATE` | off | Set to `1` to merge near-duplicate notes on compaction. Merges can be undone. |
| `KANERVA_LEARN` | off | Set to `1` to learn from which notes were used. |
| `KANERVA_WRAPUP` | on | Set to `0` to stop asking your agent for a note at the end of a session. |

### MiniLM in the plugins

`KANERVA_EMBEDDER=minilm` runs [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) on ONNX Runtime, with no torch. Its vectors match the ones behind the "Kanerva, full" column, at a cosine above 0.9999 on samples of the benchmark texts, so the same relevance model applies.

With uv the packages install themselves. With plain `python3`, run `pip install onnxruntime tokenizers` first. The 90 MB model comes from the official Hugging Face repository at a pinned revision, in the background, and search stays lexical until it arrives.

Notes written before that have no vectors, and notes from the static embedder have the wrong kind. Search still finds them by their words. To give them vectors, run this once with the setting on:

```bash
KANERVA_EMBEDDER=minilm uvx --from 'kanerva[onnx] @ git+https://github.com/katolikov/kanerva' kanerva reembed
```

### Session wrap-up

Only your agent knows what it tried and why, so Kanerva asks it. When a session has done real work (three file edits, one edit and a test run, or twelve shell commands), the agent gets one more turn to save a short outcome with `memory_remember`: what was tried, what worked and why. Kanerva still calls no model, but your agent's turn costs what any turn costs, and on Claude Code that can be a few model calls, because it has to load the tool first.

The note goes through the same secret filter and ranking as any other. A session is asked once, and again only after 20 more edits. A note the agent writes by itself restarts the count.

This has been run on Claude Code, OpenCode and Gemini CLI. Codex, Copilot CLI, Qwen Code, Devin and Cursor follow their hook docs and have not been run. The other agents are not asked yet. `opencode run` exits before the extra turn starts, so on OpenCode it needs the TUI or a server. Agents with the plugin's `Stop` hook now wait for it at the end of each turn, which takes about 0.2 s.

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
