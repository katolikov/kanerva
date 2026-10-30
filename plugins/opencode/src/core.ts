// Logic shared by the OpenCode V1 and V2 entry points.
import { spawn } from "node:child_process"
import { accessSync, constants, existsSync } from "node:fs"
import { delimiter, dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

export type Runner = (event: string, payload: Record<string, unknown>) => Promise<string>

export interface Launch {
  command: string[]
  environment: Record<string, string>
}

const PACKAGE_ROOT = dirname(dirname(fileURLToPath(import.meta.url)))
const BUNDLED = join(PACKAGE_ROOT, "lib")
const DEFAULT_TIMEOUT_MS = 20_000

function onPath(name: string, env: NodeJS.ProcessEnv): boolean {
  return (env.PATH ?? "").split(delimiter).some((dir) => {
    try {
      accessSync(join(dir, name), constants.X_OK)
      return true
    } catch {
      return false
    }
  })
}

/** How to start Kanerva: `KANERVA_COMMAND`, an installed CLI, or the bundled copy through uv or python3. */
export function resolveLaunch(env: NodeJS.ProcessEnv = process.env, bundled: string = BUNDLED): Launch {
  if (env.KANERVA_COMMAND) return { command: env.KANERVA_COMMAND.split(" "), environment: {} }
  if (onPath("kanerva", env)) return { command: ["kanerva"], environment: {} }
  const pythonPath = existsSync(bundled) ? [bundled, env.PYTHONPATH].filter(Boolean).join(delimiter) : (env.PYTHONPATH ?? "")
  const environment: Record<string, string> = pythonPath ? { PYTHONPATH: pythonPath } : {}
  if (onPath("uv", env)) {
    const packages = ["--with", "numpy>=1.26"]
    if (env.KANERVA_EMBEDDER === "static") packages.push("--with", "model2vec>=0.6")
    return { command: ["uv", "run", "--quiet", "--no-project", ...packages, "python", "-m", "kanerva"], environment }
  }
  return { command: ["python3", "-m", "kanerva"], environment }
}

/** Runs one hook event in a child process. Memory must never block the agent, so failures and timeouts return "". */
export const cliRunner =
  (launch: Launch = resolveLaunch(), timeoutMs = Number(process.env.KANERVA_TIMEOUT_MS ?? DEFAULT_TIMEOUT_MS)): Runner =>
  (event, payload) =>
    new Promise((resolve) => {
      const [program, ...args] = launch.command
      const child = spawn(program, [...args, "hook-raw", event], {
        stdio: ["pipe", "pipe", "ignore"],
        env: { ...process.env, ...launch.environment },
      })
      const timer = setTimeout(() => child.kill(), timeoutMs)
      let out = ""
      child.stdout.on("data", (chunk) => (out += chunk))
      child.on("error", () => {
        clearTimeout(timer)
        resolve("")
      })
      child.on("close", (code) => {
        clearTimeout(timer)
        resolve(code === 0 ? out : "")
      })
      child.stdin.on("error", () => {})
      child.stdin.end(JSON.stringify(payload))
    })

/** Per-session state. The context retrieved for a prompt is re-sent on each model call of that prompt. */
export class Bridge {
  private started = new Map<string, Promise<void>>()
  private startContext = new Map<string, string>()
  private promptContext = new Map<string, string>()
  private queue: Promise<unknown> = Promise.resolve()
  private run: Runner
  private cwd: string

  constructor(run: Runner, cwd: string) {
    this.run = run
    this.cwd = cwd
  }

  private payload(sessionID: string, extra: Record<string, unknown> = {}) {
    return { session_id: sessionID, cwd: this.cwd, ...extra }
  }

  /** Writes run one at a time in the background, so that tool calls never wait on memory. */
  private later(task: () => Promise<unknown>) {
    this.queue = this.queue.then(task).catch(() => {})
  }

  /** Waits for queued writes. */
  flush(): Promise<unknown> {
    return this.queue
  }

  /** Safe to call more than once: a resumed session gets no `session.created`, so a prompt also starts it. */
  sessionStart(sessionID: string): Promise<void> {
    let started = this.started.get(sessionID)
    if (!started) {
      started = this.run("session_start", this.payload(sessionID, { source: "startup" })).then((text) => {
        this.startContext.set(sessionID, text)
      })
      this.started.set(sessionID, started)
    }
    return started
  }

  async prompt(sessionID: string, prompt: string) {
    const [text] = await Promise.all([this.run("prompt", this.payload(sessionID, { prompt })), this.sessionStart(sessionID)])
    this.promptContext.set(sessionID, text)
  }

  /** Text to append to the system prompt of an outgoing model request. */
  system(sessionID: string): string {
    const parts = [this.startContext.get(sessionID), this.promptContext.get(sessionID)]
    return parts.filter(Boolean).join("\n")
  }

  toolAfter(sessionID: string, tool: string, input: unknown, output: unknown) {
    this.later(() => this.run("tool", this.payload(sessionID, { tool_name: tool, tool_input: input ?? {}, tool_response: output ?? {} })))
  }

  idle(sessionID: string) {
    this.later(() => this.run("stop", this.payload(sessionID)))
  }

  compacting(sessionID: string) {
    this.later(() => this.run("compact", this.payload(sessionID)))
  }

  compacted(sessionID: string, summary: string) {
    if (summary) this.later(() => this.run("compact_summary", this.payload(sessionID, { compact_summary: summary })))
  }

  /** Drops the state of a deleted session. */
  close(sessionID: string) {
    this.started.delete(sessionID)
    this.startContext.delete(sessionID)
    this.promptContext.delete(sessionID)
  }
}

export function promptText(parts: Array<{ type?: string; text?: string }> = []): string {
  return parts
    .filter((p) => p.type === "text" && p.text)
    .map((p) => p.text)
    .join("\n")
}

/** The text of the newest compaction summary in a V1 message list. */
export function summaryText(messages: Array<{ info?: { summary?: boolean }; parts?: Array<{ type?: string; text?: string }> }> = []): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].info?.summary) return promptText(messages[i].parts)
  }
  return ""
}
