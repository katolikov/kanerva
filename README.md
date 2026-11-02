<p align="center">
  <img src="docs/assets/banner.png" alt="Kanerva: long-term memory for coding agents" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-a8431d" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-b98a5e" alt="Python 3.10+">
  <a href="https://doi.org/10.5281/zenodo.23849910"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23849910-b98a5e" alt="DOI: 10.5281/zenodo.23849910"></a>
  <a href="https://github.com/katolikov/kanerva/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/katolikov/kanerva/ci.yml?branch=main&label=tests&color=5f7f1a" alt="Tests"></a>
</p>

Coding agents start every session from zero. You explain the test command again, the deploy steps again, the bug you already fixed last week. Memory tools help, but most of them paste the top few similar notes into your context, whatever those notes cost and whether or not they are still true.

**Kanerva gives your agent long-term memory that runs on your machine, makes no LLM calls, and injects only what the current step needs, under a hard token budget.**

- **Small context.** You set the budget in tokens, and Kanerva picks the set of notes worth the most per token. Plugins estimate tokens instead of counting them exactly, so treat the limit as close, not exact.
- **Facts stay current.** Give a fact a key, or write a near-identical note, and the new version replaces the old one. Without a key, `memory_remember` shows your agent the similar notes, and it can retire one with `memory_forget` and `replaced_by`. Old versions stay in history, and every note shows its date, newest first.
- **Quiet when it has nothing.** A gate skips prompts the store looks unable to answer, so unrelated work gets no notes.
- **Notes about the file at hand.** When a session first opens a file, the notes that name it come along.
- **Private by default.** One SQLite file in your home folder. No account, no cloud, no API key. Known key and token formats are redacted before anything is stored. The only download is the embedding model and its runtime, once, and you can turn that off.
- **No LLM of its own.** Nothing calls a model on write or read, so memory costs you the tokens it injects, plus one short turn of your own agent when a session with real work ends. You can [turn that off](#session-wrap-up).
- **Asks your agent what helped.** At the end of a session with real work, your agent says which notes helped and which misled it. Kanerva records that. Acting on it did not pass its test, so that stays opt-in.
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

"Run" means a real session against a local stub model, with a seeded note showing up in the model's request. "Docs only" means the setup follows the agent's documentation and has not been run, so please report rough edges. Each agent's README under [`plugins/`](plugins) has its limits.

The table is about notes on prompts. Notes about files were checked on Claude Code with a real model, and on Copilot CLI with a stub model.

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
uv tool install 'kanerva[onnx] @ git+https://github.com/katolikov/kanerva'
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

Search engines find text that looks like your query. Kanerva also asks what is worth paying for: a small gradient-boosted model estimates how likely each candidate is to be needed, and selection packs the most expected value into your token budget. Search runs on words and on MiniLM vectors, and the model ships as a JSON file, so there is no training on your side. Each project gets its own memory, and a `global` scope is shared across projects.

- **Changed facts.** A note with a key closes the older note with that key. Without a key, a newer near-twin replaces an older fact if it covers your question at least as well, and your agent fixes what the rule cannot tell.
- **The gate.** A small model looks at the result list and decides whether the store can answer this prompt at all. If not, the prompt gets no notes.
- **Notes about files.** When a session first reads or edits a file, it gets the notes that name that file and that the prompt has not already brought.
- **Learning.** With `KANERVA_LEARN=auto`, your agent's verdicts raise or lower a note, its kind of note in the project, and that kind across projects, and a project moves its gate and weight floor. A shadow check decides when lift starts. The model itself never changes. This is off by default, because it [did not pass its test](#benchmarks).

The name comes from Pentti Kanerva, whose sparse distributed memory showed how a few active bits out of many can recall what you need. It is also the idea behind the logo.

## Benchmarks

<!-- results:start -->
<img src="docs/assets/recall.png" alt="Evidence recovered at each token budget on the main test and the BEAM check" width="100%">

Share of the needed evidence that made it into the context, with no LLM in the loop. "Lexical" is what runs until the model arrives. Unless noted, these figures come from version 0.5.0.

| Dataset | Budget | Kanerva | Lexical | Kanerva 0.4.1 | Hybrid + reranker | Hybrid | BM25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Code history | 256 | **0.686** | 0.659 | 0.661 | 0.658 | 0.667 | 0.640 |
| Code history | 1024 | **0.804** | 0.777 | 0.782 | 0.781 | 0.789 | 0.774 |
| Code history | 4096 | **0.925** | 0.864 | 0.909 | 0.899 | 0.899 | 0.860 |
| Long conversations | 256 | **0.462** | 0.390 | 0.447 | n/a | 0.427 | 0.371 |
| Long conversations | 1024 | **0.635** | 0.550 | **0.635** | n/a | 0.430 | 0.369 |
| Long conversations | 4096 | 0.760 | 0.650 | **0.762** | n/a | 0.528 | 0.433 |

- **Code.** On 2,148 queries over 15 repositories in Python, TypeScript, Go and Rust that nothing was tuned on, Kanerva leads hybrid search by 2.4 points over all budgets, on all 15, and the reranker by 2.6.
- **C and C++.** On six more repositories, run once on the current version, Kanerva leads clearly on C, 0.939 against 0.914 for hybrid search. On C++, with templates and class methods among the facts, it leads on tasks (0.414 against 0.351) but trails overall (0.816 against 0.827), mostly because a location note gives way to a newer note holding the same value, so the value still reaches the context. An earlier test of six other repositories gave a lead of 1.1 points on all 6. Notes about files add 7.7 points at equal tokens. Working CMake, make and Ninja commands are kept as notes.
- **Long conversations.** 0.574 against 0.440 for hybrid search. This is a check and not a test, because earlier tests saw the data.
- **Stale facts.** With keys, no outdated version gets in. Without keys at 1024 tokens, the current version injects one 16.4% of the time on C and C++ and 21.0% on the earlier test's repositories, against about 98.0% for hybrid search.
- **Changed notes.** `memory_remember` lists similar notes and your agent fixes the older one. Across 104 real Claude Code sessions, it fixed 10 of 12 facts in its own conversation notes, 10 of 12 in long turns and 28 of 28 in code. It sometimes retires more than it should, and one such note survived only as history.
- **Learning.** Over 60 real sessions on repositories nothing had seen, your agent's verdicts on notes were right 96.7% of the time, against 29.2% for matching identifiers in tool calls. Acting on those verdicts did not pass its test: on six more repositories, sessions a note helped rose from 94.4% to 97.2%, but right answers fell from 76.4% to 66.7%. So by default Kanerva only records verdicts.
- **Where it falls short.** Asked in plain words with no identifier, the shipped gate brings the needed note on only 97 of 336 questions, though it reaches the candidate list on 301 of them. Lexical search stays close to BM25 from 512 tokens up. Without keys it gives up a little on exact facts. On answers from a small local model, better recall barely helped and no difference survived correction. On 603 real prompts of one developer, the gate gave notes to 53.6% of a project's own prompts and 27.7% of other projects' prompts, far weaker than on benchmarks. RepoMem queries come from templates, and real prompts are vaguer.

A query over 10,000 notes takes about 287 ms, or 0.72 s for a whole hook, which starts a new process.
<!-- results:end -->

The full method, proofs and all results are in the [paper](paper/paper.pdf).

## Configuration

Everything is optional.

| Variable | Default | What it does |
|---|---|---|
| `KANERVA_EMBEDDER` | `minilm` | `off` stays lexical and downloads nothing. `multilingual` finds notes across languages. `static` uses [model2vec](https://github.com/MinishLab/model2vec). |
| `KANERVA_PROMPT_BUDGET` | `600` | Tokens injected per prompt. |
| `KANERVA_SESSION_BUDGET` | `400` | Tokens injected at session start. |
| `KANERVA_FILE_BUDGET` | `150` | Tokens of notes per file. `0` turns file notes off. |
| `KANERVA_GATE` | on | `0` injects on every prompt. |
| `KANERVA_MIN_WEIGHT` | `0.02` | Skip notes below this estimated probability of being needed. |
| `KANERVA_HOME` | `~/.kanerva` | Where the database lives. |
| `KANERVA_CONSOLIDATE` | off | `1` merges near-duplicate notes on compaction. Merges can be undone. |
| `KANERVA_LEARN` | `shadow` | `shadow` records your agent's verdicts and acts on nothing. `auto` acts on them once a shadow check allows it, and shows a wider list on some prompts. `1` forces that, and `0` stops asking for verdicts. |
| `KANERVA_WRAPUP` | on | `0` stops asking your agent for a note at the end of a session. It is off in `claude -p` and the Agent SDK unless set to `1`. |

### MiniLM

[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) runs on ONNX Runtime, with no torch. With uv you do nothing: the 90 MB model and ONNX Runtime arrive in the background on first use, search stays on words until then, and no hook waits for them. Without uv, run `pip install onnxruntime tokenizers`. After switching embedders, give older notes vectors with:

```bash
uvx --from 'kanerva[onnx] @ git+https://github.com/katolikov/kanerva' kanerva reembed
```

### Other languages

Russian notes get a stemmer and stopwords with no setting. `KANERVA_EMBEDDER=multilingual` lets a Russian question find English notes and the other way round, at the cost of a larger, slower model.

### Session wrap-up

Only your agent knows what it tried and why, so when a session has done real work it gets one more turn to say which notes helped with `memory_feedback` and to save a short outcome with `memory_remember`. Kanerva calls no model, but that turn costs what any turn costs. It has been run on Claude Code, Codex, OpenCode and Gemini CLI.

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
  doi    = {10.5281/zenodo.23849910},
  note   = {Preprint}
}
```

## License

[Apache-2.0](LICENSE). The LoCoMo and BEAM benchmarks have their own licenses and are downloaded when you run the experiments, not shipped here.
