// Pi extension. Notes go in as a hidden message before each run; edits, commands and compaction are recorded in the background.
import { Bridge, type Runner, cliRunner } from "./core.ts"

export function attach(pi: any, run: Runner) {
  const bridges = new Map<string, Bridge>()
  const bridgeFor = (ctx: any) => {
    let bridge = bridges.get(ctx.cwd)
    if (!bridge) {
      bridge = new Bridge(run, ctx.cwd)
      bridges.set(ctx.cwd, bridge)
    }
    return bridge
  }
  const sessionOf = (ctx: any): string => ctx.sessionManager?.getSessionId?.() ?? "default"

  pi.on("before_agent_start", async (event: any, ctx: any) => {
    const bridge = bridgeFor(ctx)
    const session = sessionOf(ctx)
    await bridge.prompt(session, event.prompt ?? "")
    const content = bridge.system(session)
    return content ? { message: { customType: "kanerva", content, display: false } } : undefined
  })
  pi.on("tool_result", (event: any, ctx: any) => {
    bridgeFor(ctx).toolAfter(sessionOf(ctx), event.toolName, event.input, { exit_code: event.isError ? 1 : 0 })
  })
  pi.on("agent_end", (_event: any, ctx: any) => {
    bridgeFor(ctx).idle(sessionOf(ctx))
  })
  pi.on("session_before_compact", (_event: any, ctx: any) => {
    bridgeFor(ctx).compacting(sessionOf(ctx))
  })
  pi.on("session_compact", (event: any, ctx: any) => {
    bridgeFor(ctx).compacted(sessionOf(ctx), event.compactionEntry?.summary ?? "")
  })
  pi.on("session_shutdown", async (_event: any, ctx: any) => {
    const bridge = bridgeFor(ctx)
    await bridge.flush()
    bridge.close(sessionOf(ctx))
  })
}

export default (pi: any) => attach(pi, cliRunner())
