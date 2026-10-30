// OpenCode entry point. One default export serves both plugin generations: V2 reads `id` and `setup`,
// and V1 (1.18.29 or newer) reads `server`. Memory tools come from the MCP server; see the README.
import { Bridge, type Runner, promptText, pythonRunner } from "./core.ts"

/** Event and hook field names follow the types of @opencode/plugin 2.2.3. */
export async function setupV2(ctx: any, run: Runner) {
  const bridge = new Bridge(run, ctx.location.directory)
  await ctx.session.hook("prompt", async (event: any) => {
    await bridge.prompt(event.sessionID, event.prompt?.text ?? "")
  })
  await ctx.session.hook("context", (event: any) => {
    const text = bridge.system(event.sessionID)
    if (text) event.system.push({ type: "text", text })
  })
  await ctx.session.hook("compaction", async (event: any) => {
    await bridge.compacting(event.sessionID)
  })
  await ctx.tool.hook("execute.after", async (event: any) => {
    if (event.status === "completed") await bridge.toolAfter(event.sessionID, event.tool, event.input, event.result)
  })
  const controller = new AbortController()
  void (async () => {
    for await (const event of ctx.event.subscribe({
      types: ["session.created", "session.idle"],
      signal: controller.signal,
    })) {
      const sessionID = event.data?.sessionID
      if (event.type === "session.created") await bridge.sessionStart(sessionID)
      if (event.type === "session.idle") await bridge.idle(sessionID)
    }
  })()
  return () => controller.abort()
}

const v2 = {
  id: "kanerva",
  setup: (ctx: any) => setupV2(ctx, pythonRunner()),
}

const server = async ({ directory }: { directory: string }) => {
  const bridge = new Bridge(pythonRunner(), directory)
  return {
    event: async ({ event }: any) => {
      const sessionID = event.properties?.sessionID ?? event.properties?.info?.id
      if (event.type === "session.created") await bridge.sessionStart(sessionID)
      if (event.type === "session.idle") await bridge.idle(sessionID)
    },
    "chat.message": async (input: any, output: any) => {
      await bridge.prompt(input.sessionID, promptText(output.parts))
    },
    "experimental.chat.system.transform": async (input: any, output: any) => {
      const text = input.sessionID ? bridge.system(input.sessionID) : ""
      if (text) output.system.push(text)
    },
    "tool.execute.after": async (input: any, output: any) => {
      await bridge.toolAfter(input.sessionID, input.tool, input.args, { stdout: output.output })
    },
    "experimental.session.compacting": async (input: any) => {
      await bridge.compacting(input.sessionID)
    },
  }
}

export default { ...v2, server }
