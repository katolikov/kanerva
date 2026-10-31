// OpenClaw entry point. Notes are prepended to each prompt; edits, commands and compaction are recorded in the background.
import { Bridge, type Runner, cliRunner, resolveLaunch } from "./core.ts"

const HOOK_TIMEOUT_MS = 12_000

export function attach(api: any, run: Runner) {
  const bridges = new Map<string, Bridge>()
  const bridgeFor = (ctx: any) => {
    const directory = ctx.workspaceDir ?? process.cwd()
    let bridge = bridges.get(directory)
    if (!bridge) {
      bridge = new Bridge(run, directory)
      bridges.set(directory, bridge)
    }
    return bridge
  }
  const sessionOf = (ctx: any): string => ctx.sessionId ?? ctx.sessionKey ?? "default"

  api.on("before_prompt_build", async (event: any, ctx: any) => {
    const bridge = bridgeFor(ctx)
    const session = sessionOf(ctx)
    await bridge.prompt(session, event.currentUserMessage ?? event.prompt ?? "")
    const text = bridge.system(session)
    return text ? { prependContext: text } : undefined
  })
  api.on("after_tool_call", (event: any, ctx: any) => {
    bridgeFor(ctx).toolAfter(sessionOf(ctx), event.toolName, event.params, { stderr: event.error ?? "", exit_code: event.error ? 1 : 0 })
  })
  api.on("agent_end", (_event: any, ctx: any) => {
    bridgeFor(ctx).idle(sessionOf(ctx))
  })
  api.on("before_compaction", (_event: any, ctx: any) => {
    bridgeFor(ctx).compacting(sessionOf(ctx))
  })
  api.on("session_end", (_event: any, ctx: any) => {
    bridgeFor(ctx).close(sessionOf(ctx))
  })
}

export default {
  id: "kanerva",
  name: "Kanerva memory",
  description: "Long-term memory for coding agents: local, no LLM calls, hard token budget.",
  register: (api: any) => attach(api, cliRunner(resolveLaunch(), HOOK_TIMEOUT_MS)),
}
