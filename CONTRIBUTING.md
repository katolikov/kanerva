# Contributing

Thanks for helping out. Bug reports, plugin fixes and benchmark results are all welcome.

## Setup

```bash
make install
make test
```

`make test` runs the Python tests and the OpenCode plugin tests.

## Ground rules

- The memory layer stays local. It must not call a paid API or need an LLM.
- Experiments use the library as shipped, so if a method needs a change, make it in `src/kanerva`.
- After changing `src/kanerva`, run `scripts/sync_plugin.sh` so that both plugins carry the same code.
- A retrieval change should come with before and after numbers, including where it gets worse.
- Tune on development data only.
- Numbers in the paper come from `results/`. `make check` enforces this.

## Benchmarks

```bash
make data        # downloads the datasets
make reproduce   # runs everything
```

LoCoMo is CC BY-NC 4.0 and BEAM is CC BY-SA 4.0. They are downloaded at run time and are not part of this repository.
