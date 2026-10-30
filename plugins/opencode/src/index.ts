// OpenCode entry point. One default export serves both plugin generations: V2 reads `id` and `setup`,
// and V1 (1.18.29 or newer) reads `server`.
import { Bridge, type Launch, type Runner, cliRunner, promptText, resolveLaunch, summaryText } from "./core.ts"

const SERVER = "kanerva"

export async function setupV2(ctx: any, run: Runner, launch: Launch) {
  const bridge = new Bridge(run, ctx.location.directory)
  await ctx.mcp.transform((editor: any) => {
    // codemode off: the model calls memory_search directly instead of scripting it inside `execute`.
    editor.set(SERVER, { type: "local", command: [...launch.command, "mcp"], environment: launch.environment, codemode: false })
  })
  await ctx.session.hook("prompt", async (event: any) => {
    await bridge.prompt(event.sessionID, event.prompt?.text ?? "")
  })
  await ctx.session.hook("context", (event: any) => {
    const text = bridge.system(event.sessionID)
    if (text) event.system.push({ type: "text", text })
  })
  await ctx.session.hook("compaction", (event: any) => {
    bridge.compacting(event.sessionID)
  })
  await ctx.tool.hook("execute.after", (event: any) => {
    if (event.status === "completed") bridge.toolAfter(event.sessionID, event.tool, event.input, event.result)
  })
  const controller = new AbortController()
  void (async () => {
    const events = ctx.event.subscribe({
      types: ["session.created", "session.idle", "session.deleted", "session.compaction.ended"],
      signal: controller.signal,
    })
    for await (const event of events) {
      const sessionID = event.data?.sessionID
      if (!sessionID) continue
      if (event.type === "session.created") await bridge.sessionStart(sessionID)
      if (event.type === "session.idle") bridge.idle(sessionID)
      if (event.type === "session.deleted") bridge.close(sessionID)
      if (event.type === "session.compaction.ended") bridge.compacted(sessionID, event.data.summary ?? event.data.text ?? "")
    }
  })().catch(() => {})
  return async () => {
    controller.abort()
    await bridge.flush()
  }
}

export async function serverV1({ client, directory }: any, run: Runner, launch: Launch) {
  const bridge = new Bridge(run, directory)
  return {
    config: async (config: any) => {
      config.mcp ??= {}
      config.mcp[SERVER] ??= { type: "local", command: [...launch.command, "mcp"], environment: launch.environment, enabled: true }
    },
    event: async ({ event }: any) => {
      const properties = event.properties ?? {}
      const sessionID = properties.sessionID ?? properties.info?.id
      if (!sessionID) return
      if (event.type === "session.created") await bridge.sessionStart(sessionID)
      if (event.type === "session.idle") bridge.idle(sessionID)
      if (event.type === "session.deleted") bridge.close(sessionID)
      if (event.type === "session.compacted") {
        const messages = await client?.session?.messages?.({ path: { id: sessionID } }).catch(() => undefined)
        bridge.compacted(sessionID, summaryText(messages?.data))
      }
    },
    "chat.message": async (input: any, output: any) => {
      await bridge.prompt(input.sessionID, promptText(output.parts))
    },
    "experimental.chat.system.transform": async (input: any, output: any) => {
      const text = input.sessionID ? bridge.system(input.sessionID) : ""
      if (text) output.system.push(text)
    },
    "tool.execute.after": async (input: any, output: any) => {
      bridge.toolAfter(input.sessionID, input.tool, input.args, { stdout: output.output })
    },
    "experimental.session.compacting": async (input: any) => {
      bridge.compacting(input.sessionID)
    },
    dispose: async () => {
      await bridge.flush()
    },
  }
}

const launch = resolveLaunch()

export default {
  id: "kanerva",
  setup: (ctx: any) => setupV2(ctx, cliRunner(launch), launch),
  server: (input: any) => serverV1(input, cliRunner(launch), launch),
}
