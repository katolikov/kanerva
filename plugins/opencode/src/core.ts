// Host-neutral logic shared by the OpenCode V1 and V2 entry points. `Runner` calls the kanerva CLI, so tests can inject a fake.
import { spawn } from "node:child_process"

export type Runner = (event: string, payload: Record<string, unknown>) => Promise<string>

export const pythonRunner =
  (command: string[] = (process.env.KANERVA_COMMAND ?? "kanerva").split(" ")): Runner =>
  (event, payload) =>
    new Promise((resolve) => {
      const child = spawn(command[0], [...command.slice(1), "hook-raw", event], { stdio: ["pipe", "pipe", "ignore"] })
      let out = ""
      child.stdout.on("data", (chunk) => (out += chunk))
      child.on("error", () => resolve(""))
      child.on("close", () => resolve(out))
      child.stdin.on("error", () => {})
      child.stdin.end(JSON.stringify(payload))
    })

/** Per-session state: the context retrieved for the latest prompt is re-sent on each model call of that prompt. */
export class Bridge {
  private startContext = new Map<string, string>()
  private promptContext = new Map<string, string>()
  private run: Runner
  private cwd: string

  constructor(run: Runner, cwd: string) {
    this.run = run
    this.cwd = cwd
  }

  private payload(sessionID: string, extra: Record<string, unknown> = {}) {
    return { session_id: sessionID, cwd: this.cwd, ...extra }
  }

  async sessionStart(sessionID: string) {
    this.startContext.set(sessionID, await this.run("session_start", this.payload(sessionID, { source: "startup" })))
  }

  async prompt(sessionID: string, prompt: string) {
    this.promptContext.set(sessionID, await this.run("prompt", this.payload(sessionID, { prompt })))
  }

  /** Text to append to the system prompt of an outgoing model request. */
  system(sessionID: string): string {
    const parts = [this.startContext.get(sessionID), this.promptContext.get(sessionID)]
    return parts.filter(Boolean).join("\n")
  }

  async toolAfter(sessionID: string, tool: string, input: unknown, output: unknown) {
    await this.run("tool", this.payload(sessionID, { tool_name: tool, tool_input: input ?? {}, tool_response: output ?? {} }))
  }

  async idle(sessionID: string) {
    await this.run("stop", this.payload(sessionID))
  }

  async compacting(sessionID: string) {
    await this.run("compact", this.payload(sessionID))
  }

  async compacted(sessionID: string, summary: string) {
    if (summary) await this.run("compact_summary", this.payload(sessionID, { compact_summary: summary }))
  }
}

export function promptText(parts: Array<{ type?: string; text?: string }>): string {
  return parts
    .filter((p) => p.type === "text" && p.text)
    .map((p) => p.text)
    .join("\n")
}
