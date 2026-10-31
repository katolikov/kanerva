# Security

Kanerva stores what your agent sees in a local SQLite file, so a leak there matters. If you find a way to get a secret past the redaction filter, or any other vulnerability, please use [private vulnerability reporting](https://github.com/katolikov/kanerva/security/advisories/new) instead of a public issue.

## What Kanerva does

- Runs locally. The memory layer sends nothing anywhere. Its only network call is one download of the MiniLM model from the official Hugging Face repository, at a pinned revision with checked hashes. `KANERVA_EMBEDDER=off` turns that off, and `multilingual` or `static` download their own model instead.
- Redacts API keys, tokens, private keys, JWTs, `Authorization` headers, credentials in URLs and values assigned to names such as `DB_PASSWORD` before anything is written to disk, including prompts and commands kept for the session. See `src/kanerva/secrets_filter.py`.
- Keeps everything in `~/.kanerva/memory.sqlite`, which you can inspect, export or delete at any time.

Redaction is pattern-based and cannot catch everything, so avoid pasting secrets into prompts in the first place.
