// Cline plugin. Notes are added at the start of each run; edits, commands and run ends are recorded in the background.
import { Bridge, type Runner, cliRunner, resolveLaunch } from "./core.ts"

/** The newest user text of a run, without the tag Cline wraps it in. */
export function latestPrompt(messages: any[] = []): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role !== "user") continue
    const content = messages[i].content
    const text = typeof content === "string" ? content : (content ?? []).filter((part: any) => part.type === "text").map((part: any) => part.text).join("\n")
    return text.replace(/<\/?user_input[^>]*>/g, "").trim()
  }
  return ""
}

export function createPlugin(run: Runner) {
  let bridge: Bridge | undefined
  let session = "default"
  return {
    name: "kanerva",
    manifest: { capabilities: ["hooks"] },
    setup(_api: any, ctx: any) {
      bridge = new Bridge(run, ctx.cwd ?? ctx.workspaceInfo?.rootPath ?? process.cwd())
      session = ctx.session?.sessionId ?? session
    },
    hooks: {
      async beforeRun(context: any) {
        if (!bridge) return undefined
        await bridge.prompt(session, latestPrompt(context.snapshot?.messages))
        const text = bridge.system(session)
        return text ? { appendContext: text } : undefined
      },
      afterTool(context: any) {
        const name = context.toolCall?.name ?? context.tool?.name ?? ""
        const response = { exit_code: context.result?.isError ? 1 : 0 }
        if (name === "run_commands") {
          for (const command of context.input?.commands ?? []) bridge?.toolAfter(session, "bash", { command: String(command) }, response)
        } else {
          bridge?.toolAfter(session, name, context.input, response)
        }
        return undefined
      },
      afterRun() {
        bridge?.idle(session)
      },
    },
  }
}

export default createPlugin(cliRunner(resolveLaunch()))
