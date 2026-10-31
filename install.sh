#!/bin/sh
# Installs Kanerva for one agent, or for every supported agent found on this machine.
#   curl -fsSL https://raw.githubusercontent.com/katolikov/kanerva/main/install.sh | sh -s -- opencode
set -eu

REPO="${KANERVA_REPO:-https://github.com/katolikov/kanerva.git}"
SOURCE="${KANERVA_SOURCE:-katolikov/kanerva}"
SRC="${KANERVA_SRC:-$HOME/.kanerva/src}"
RUN="$SRC/plugins/claude-code/scripts/run.sh"
TARGET="${1:-all}"
AGENTS="claude opencode codex copilot openclaw pi cursor hermes qwen devin gemini antigravity amp kiro cline"

fetch() {
  command -v git >/dev/null 2>&1 || { echo "kanerva: git is required" >&2; exit 1; }
  if [ -d "$SRC/.git" ]; then
    git -C "$SRC" pull --quiet --ff-only
  else
    mkdir -p "$(dirname "$SRC")"
    git clone --quiet --depth 1 "$REPO" "$SRC"
  fi
}

check_python() {
  if ! command -v uv >/dev/null 2>&1 && ! python3 -c "import numpy" >/dev/null 2>&1; then
    echo "kanerva: install uv (https://docs.astral.sh/uv/) or NumPy for python3 first" >&2
    exit 1
  fi
}

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "kanerva: the $1 command was not found" >&2; exit 1; }
}

add_config() {
  if command -v python3 >/dev/null 2>&1; then
    python3 "$SRC/scripts/add_to_config.py" "$SRC/plugins/$1" "$2" "$RUN"
  else
    uv run --quiet --no-project python "$SRC/scripts/add_to_config.py" "$SRC/plugins/$1" "$2" "$RUN"
  fi
}

install_opencode() {
  fetch
  dir="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/plugins"
  mkdir -p "$dir"
  printf 'export { default } from "%s/plugins/opencode/src/index.ts"\n' "$SRC" > "$dir/kanerva.ts"
  echo "Kanerva plugin added to $dir. Restart OpenCode to load it."
}

install_claude() {
  need claude
  claude plugin marketplace add "$SOURCE"
  claude plugin install kanerva@kanerva
  echo "Kanerva plugin installed. Restart Claude Code to load it."
}

install_codex() {
  need codex
  codex plugin marketplace add "$SOURCE"
  codex plugin add kanerva@kanerva
  echo "Kanerva plugin installed. Start Codex, run /hooks and trust the Kanerva hooks, or they stay off."
}

install_copilot() {
  need copilot
  copilot plugin marketplace add "$SOURCE"
  copilot plugin install kanerva@kanerva
  echo "Kanerva plugin installed. Restart Copilot CLI to load it."
}

install_openclaw() {
  need openclaw
  fetch
  openclaw plugins install "$SRC/plugins/openclaw" --link --force --accept-capabilities
  openclaw config set plugins.entries.kanerva.hooks.allowConversationAccess true --strict-json
  openclaw mcp set kanerva "{\"command\":\"$RUN\",\"args\":[\"mcp\"]}"
  echo "Kanerva plugin linked from $SRC/plugins/openclaw."
}

install_pi() {
  need pi
  fetch
  pi install "$SRC"
  pi mcp add memory --exposure direct -- "$RUN" mcp
  echo "Kanerva extension added. Run /reload in Pi, or restart it."
}

install_cursor() {
  fetch
  add_config cursor/hooks.json "$HOME/.cursor/hooks.json"
  add_config cursor/mcp.json "$HOME/.cursor/mcp.json"
  echo "Kanerva added to Cursor. Restart Cursor to load it."
}

install_hermes() {
  need hermes
  hermes plugins install "$SOURCE/plugins/hermes" --enable --force
  echo "Kanerva plugin installed. For the memory_* tools, see plugins/hermes/README.md."
}

install_qwen() {
  need qwen
  fetch
  printf 'y\ny\n' | qwen extensions install "$SRC/plugins/qwen" --consent
  echo "Kanerva extension installed. Restart Qwen Code to load it."
}

install_devin() {
  need devin
  fetch
  devin plugins install "$SOURCE#plugins/claude-code" --yes
  add_config devin/hooks.json "${XDG_CONFIG_HOME:-$HOME/.config}/devin/config.json"
  echo "Kanerva added to Devin. Restart Devin to load it."
}

install_gemini() {
  need gemini
  fetch
  printf 'y\ny\n' | gemini extensions install "$SRC/plugins/gemini" --consent --skip-settings
  echo "Kanerva extension installed. Restart Gemini CLI to load it."
}

install_antigravity() {
  fetch
  add_config antigravity/hooks.json "$HOME/.gemini/config/hooks.json"
  add_config antigravity/mcp_config.json "$HOME/.gemini/config/mcp_config.json"
  echo "Kanerva added to Antigravity. Restart it to load the hooks."
}

install_amp() {
  fetch
  dir="${XDG_CONFIG_HOME:-$HOME/.config}/amp/plugins"
  mkdir -p "$dir"
  printf 'export { default } from "%s/plugins/amp/src/index.ts"\n' "$SRC" > "$dir/kanerva.ts"
  echo "Kanerva plugin added to $dir. Reload plugins in Amp to load it."
}

install_kiro() {
  fetch
  add_config kiro/hooks.json "$HOME/.kiro/hooks/kanerva.json"
  add_config kiro/mcp.json "$HOME/.kiro/settings/mcp.json"
  echo "Kanerva added to Kiro. Restart Kiro to load it."
}

install_cline() {
  need cline
  fetch
  cline plugin install "$SRC/plugins/cline" --force
  cline mcp install kanerva --yes -- "$RUN" mcp
  echo "Kanerva plugin installed. Restart the Cline CLI to load it."
}

found() {
  case "$1" in
    antigravity) command -v agy >/dev/null 2>&1 ;;
    kiro) command -v kiro-cli >/dev/null 2>&1 || [ -d "$HOME/.kiro" ] ;;
    cursor) command -v cursor >/dev/null 2>&1 || [ -d "$HOME/.cursor" ] ;;
    *) command -v "$1" >/dev/null 2>&1 ;;
  esac
}

check_python
case " $AGENTS " in
  *" $TARGET "*) "install_$TARGET" ;;
  *)
    if [ "$TARGET" != "all" ]; then
      echo "usage: install.sh [$(echo "$AGENTS" | tr ' ' '|')|all]" >&2
      exit 2
    fi
    any=""
    for agent in $AGENTS; do
      if found "$agent"; then
        any=1
        ( "install_$agent" ) || echo "kanerva: could not install for $agent" >&2
      fi
    done
    if [ -z "$any" ]; then
      echo "kanerva: no supported agent was found" >&2
      exit 1
    fi
    ;;
esac
