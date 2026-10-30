# Kanerva

Local memory for coding agents that injects only what the current step needs, under a hard token budget, and knows when a stored fact has gone stale.

Kanerva runs on SQLite with no LLM in the loop. It works with Claude Code and OpenCode, and with any MCP client.

**Status: research prototype.** The results below are retrieval-level, measured without an LLM. The plugins have passed simulated hook tests, but have not yet run in live sessions.

## Why

Agent memory products mostly fetch the top-k similar notes. That ignores what a note costs in tokens, ignores what the other notes already say, and has no idea whether a note is still true. Kanerva treats memory as budgeted selection:

- **Selection** maximizes a monotone submodular objective under a token budget, and each query carries a certificate of its optimality gap.
- **Validity** is modelled apart from usefulness. Change rates are learned per class of fact from the store's own version history.
- **Learning** updates a small linear policy from execution signals, such as whether a note's identifiers reappeared in later tool calls.
- **Consolidation is reversible.** Summaries keep links to their sources, and nothing is deleted without an explicit command.

## Demo

```text
$ kanerva add "Tests run with make test-all, not pytest directly" --key cmd:test
$ kanerva add "Tests run with tox -e py312 since the CI migration" --key cmd:test
$ kanerva add "deploy token ghp_abcdefghijklmnopqrstuvwxyz0123456789 lives in the vault"
$ kanerva search "how do I run the tests?" --budget 60
[#3 2026-10-30] Tests run with tox -e py312 since the CI migration
[#1 2026-10-30 superseded] Tests run with make test-all, not pytest directly
$ kanerva list
[#4 2026-10-30] deploy token [REDACTED:github-token] lives in the vault
...
```

The complete run is in [docs/demo-transcript.txt](docs/demo-transcript.txt).

## Quickstart

```bash
git clone <this repository> kanerva && cd kanerva
uv tool install .
kanerva add "We deploy from the release branch" --key process:deploy
kanerva search "how do we deploy?"
```

### Claude Code

```bash
scripts/sync_plugin.sh
claude --plugin-dir ./plugins/claude-code
```

The plugin injects memory on `SessionStart` and `UserPromptSubmit`, captures edits and test commands on `PostToolUse`, resolves feedback on `Stop`, and consolidates on `PreCompact`. Skills: `/kanerva:memory`, `/kanerva:remember`, `/kanerva:forget`.

### OpenCode

```jsonc
// opencode.jsonc (V2)
{
  "plugins": ["/absolute/path/to/plugins/opencode"],
  "mcp": { "servers": { "kanerva": { "type": "local", "command": ["kanerva", "mcp"] } } }
}
```

The one package also carries the V1 entry point. V1 uses `"plugin"` and `mcp.kanerva` with `"enabled": true`.

### Library

```python
from kanerva import Memory

memory = Memory("memory.sqlite")
memory.write("The API timeout is 30 seconds", key="config:timeout")
memory.write("The API timeout is 60 seconds", key="config:timeout")  # supersedes, keeps history
result = memory.retrieve("what is the API timeout?", budget=200)
print(result.render())
memory.feedback(result.retrieval_id, {result.items[0].id: 1.0})
```

Other operations are `consolidate`, `forget` and `expand` in `kanerva.consolidate`. Hybrid retrieval needs an embedder: `pip install "kanerva[embeddings]"` and pass `Model2VecEmbedder()`, or `SentenceTransformerEmbedder()` with the `transformers` extra. Without one, Kanerva runs lexical-only, which is the plugin default.

### Where data lives

Everything is in `~/.kanerva/memory.sqlite`, or `$KANERVA_HOME`. Each project has its own scope, and a `global` scope is shared. Inspect and edit with `kanerva list`, `kanerva show <id>`, `kanerva edit <id> "..."`, `kanerva forget <id>` and `kanerva export memory.md`. A secret detector redacts credentials before anything is stored.

## Results

![Recall against tokens](paper/figs/pareto.png)

<!-- results:start -->
<!-- results:end -->

## Compared to alternatives

| | Kanerva | Claude Code files | Mem0 | Zep / Graphiti | Letta |
|---|---|---|---|---|---|
| LLM needed on write | no | model writes the notes | yes, for fact extraction | yes | agent tool calls |
| Hard token budget | yes | 200 lines or 25 KB, unranked | no, top-k | no, top-k | block sizes |
| Redundancy-aware selection | yes | no | no | no | no |
| Staleness | probabilistic and versioned | none | LLM update or delete, decay on the platform | bi-temporal edges | agent-managed |
| Learns from feedback | linear policy | no | no | no | no |
| Runs without a service | yes | yes | yes, with a local vector store | needs a graph DB | server or CLI |
| Maturity | research prototype | production | production | production | production |
| Evaluated with an LLM | **no** | n/a | yes, vendor-run | yes, vendor-run | yes, vendor-run |

Those systems have features Kanerva lacks, such as LLM extraction, graph traversal, hosted operation and large user bases. Mem0 with LLM extraction, Graphiti and Letta were not run here, because no LLM was available. See the paper's limitations section.

## Reproduce

```bash
make install && make data
make reproduce        # all experiments, with a wall-clock cap per step
make paper && make check
```

`results/*.json` holds seeds and commit hashes. `make check` confirms that every number in the paper comes from those files. The Mem0 baseline needs its own environment; see `experiments/run_systems.py`.

## Paper

[paper/paper.pdf](paper/paper.pdf)

```bibtex
@misc{kanerva2026,
  title  = {Kanerva: Token-Budgeted, Validity-Aware Memory for Coding Agents, Without an LLM in the Loop},
  author = {Katolikov, Artem},
  year   = {2026},
  note   = {Preprint}
}
```

## License

Apache-2.0. LoCoMo is CC BY-NC 4.0, so it is fetched at run time and not redistributed.
