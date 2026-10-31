// Amp plugin. Notes are added at the start of each turn; edits, commands and turn ends are recorded in the background.
import { fileURLToPath } from "node:url"
import { Bridge, type Runner, cliRunner, resolveLaunch } from "./core.ts"

export const description = "Long-term memory for coding agents: local, no LLM calls, hard token budget."

export function workspaceOf(amp: any): string {
  const root = amp.system?.workspaceRoot?.toString?.()
  return root?.startsWith("file:") ? fileURLToPath(root) : (root ?? process.cwd())
}

export function attach(amp: any, run: Runner) {
  const bridge = new Bridge(run, workspaceOf(amp))
  amp.on("agent.start", async (event: any) => {
    await bridge.prompt(event.thread.id, event.message ?? "")
    const content = bridge.system(event.thread.id)
    return content ? { message: { content, display: false } } : undefined
  })
  amp.on("tool.result", (event: any) => {
    if (event.status === "cancelled") return
    bridge.toolAfter(event.thread.id, event.tool, event.input, { stderr: event.error ?? "", exit_code: event.status === "error" ? 1 : 0 })
  })
  amp.on("agent.end", (event: any) => {
    bridge.idle(event.thread.id)
  })
}

export default (amp: any) => attach(amp, cliRunner(resolveLaunch()))
