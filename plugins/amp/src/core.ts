// Logic shared by the OpenCode V1 and V2 entry points.
import { spawn } from "node:child_process"
import { accessSync, constants, existsSync } from "node:fs"
import { delimiter, dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

export type Runner = (event: string, payload: Record<string, unknown>) => Promise<string>

/** Hands a request for an outcome note to the model, as a turn of its own. */
export type Followup = (sessionID: string, text: string) => Promise<unknown>

export interface Launch {
  command: string[]
  environment: Record<string, string>
  cwd?: string
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

/**
 * How to start Kanerva: `KANERVA_COMMAND`, an installed CLI, or the bundled copy through uv or python3.
 * The bundled copy runs from the plugin folder, so that a project's own modules and .python-version stay out of the way.
 */
export function resolveLaunch(env: NodeJS.ProcessEnv = process.env, bundled: string = BUNDLED): Launch {
  if (env.KANERVA_COMMAND) return { command: env.KANERVA_COMMAND.split(" "), environment: {} }
  if (onPath("kanerva", env)) return { command: ["kanerva"], environment: {} }
  const pythonPath = existsSync(bundled) ? [bundled, env.PYTHONPATH].filter(Boolean).join(delimiter) : (env.PYTHONPATH ?? "")
  const environment: Record<string, string> = pythonPath ? { PYTHONPATH: pythonPath } : {}
  if (onPath("uv", env)) {
    const packages = ["--with", "numpy>=1.26"]
    if (env.KANERVA_EMBEDDER === "static") packages.push("--with", "model2vec>=0.6")
    // Without a setting, the first hook must not wait for packages: a background download brings them with the model.
    const model = join(env.KANERVA_HOME ?? join(env.HOME ?? "", ".kanerva"), "models", "all-MiniLM-L6-v2", "onnx", "model.onnx")
    if (env.KANERVA_EMBEDDER === "minilm" || env.KANERVA_EMBEDDER === "multilingual" || (!env.KANERVA_EMBEDDER && existsSync(model))) {
      packages.push("--with", "onnxruntime>=1.18", "--with", "tokenizers>=0.19")
    }
    return { command: ["uv", "run", "--quiet", "--no-project", ...packages, "python", "-m", "kanerva"], environment, cwd: dirname(bundled) }
  }
  return { command: ["python3", "-m", "kanerva"], environment, cwd: dirname(bundled) }
}

/** The same launch, tied to a project. */
export function forProject(launch: Launch, directory: string): Launch {
  return { ...launch, environment: { ...launch.environment, KANERVA_PROJECT_DIR: directory } }
}

/** Runs one hook event in a child process. Memory must never block the agent, so failures and timeouts return "". */
export const cliRunner =
  (launch: Launch = resolveLaunch(), timeoutMs = Number(process.env.KANERVA_TIMEOUT_MS ?? DEFAULT_TIMEOUT_MS)): Runner =>
  (event, payload) =>
    new Promise((resolve) => {
      let settled = false
      let child: ReturnType<typeof spawn> | undefined
      const finish = (text: string) => {
        if (settled) return
        settled = true
        clearTimeout(timer)
        resolve(text)
      }
      const timer = setTimeout(() => {
        child?.kill()
        finish("")
      }, timeoutMs)
      const attempt = (command: string[], otherwise: () => void) => {
        const [program, ...args] = command
        let failed = false
        const fail = () => {
          if (failed) return
          failed = true
          otherwise()
        }
        const process_ = spawn(program, [...args, "hook-raw", event], {
          stdio: ["pipe", "pipe", "ignore"],
          env: { ...process.env, ...launch.environment },
          cwd: launch.cwd,
        })
        child = process_
        let out = ""
        process_.stdout.on("data", (chunk) => (out += chunk))
        process_.on("error", fail)
        process_.on("close", (code) => (code === 0 ? finish(out) : fail()))
        process_.stdin.on("error", () => {})
        process_.stdin.end(JSON.stringify(payload))
      }
      // Offline first: with the packages cached, memory must not wait on a network that may not answer.
      if (launch.command[0] === "uv") attempt(["uv", "run", "--offline", ...launch.command.slice(2)], () => attempt(launch.command, () => finish("")))
      else attempt(launch.command, () => finish(""))
    })

/** Per-session state. The context retrieved for a prompt is re-sent on each model call of that prompt. */
export class Bridge {
  private started = new Map<string, Promise<void>>()
  private startContext = new Map<string, string>()
  private promptContext = new Map<string, string>()
  private fileContext = new Map<string, string>()
  private subagents = new Set<string>()
  private queue: Promise<unknown> = Promise.resolve()
  private run: Runner
  private cwd: string
  private followup?: Followup

  constructor(run: Runner, cwd: string, followup?: Followup) {
    this.run = run
    this.cwd = cwd
    this.followup = followup
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
    this.fileContext.delete(sessionID)
  }

  /** Text to append to the system prompt of an outgoing model request. */
  system(sessionID: string): string {
    const parts = [this.startContext.get(sessionID), this.promptContext.get(sessionID), this.fileContext.get(sessionID)]
    return parts.filter(Boolean).join("\n")
  }

  toolAfter(sessionID: string, tool: string, input: unknown, output: unknown) {
    this.later(async () => {
      const text = await this.run("tool", this.payload(sessionID, { tool_name: tool, tool_input: input ?? {}, tool_response: output ?? {} }))
      if (text) this.fileContext.set(sessionID, [this.fileContext.get(sessionID), text].filter(Boolean).join("\n"))
    })
  }

  /** A subagent's session ends with its task, so the note is left to the session that started it. */
  subagent(sessionID: string) {
    this.subagents.add(sessionID)
  }

  idle(sessionID: string) {
    this.later(async () => {
      const text = await this.run("stop", this.payload(sessionID))
      if (text && !this.subagents.has(sessionID)) await this.followup?.(sessionID, text)
    })
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
    this.fileContext.delete(sessionID)
    this.subagents.delete(sessionID)
  }
}

export function promptText(parts: Array<{ type?: string; text?: string; synthetic?: boolean }> = []): string {
  return parts
    .filter((p) => p.type === "text" && p.text && !p.synthetic)
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
