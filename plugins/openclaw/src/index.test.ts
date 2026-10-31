import assert from "node:assert/strict"
import { test } from "node:test"
import plugin, { attach } from "./index.ts"

function host() {
  const handlers = new Map<string, Function>()
  const calls: Array<[string, Record<string, unknown>]> = []
  const run = async (event: string, payload: Record<string, unknown>) => {
    calls.push([event, payload])
    return event === "prompt" ? "note about tests" : ""
  }
  attach({ on: (name: string, handler: Function) => handlers.set(name, handler) }, run)
  return { handlers, calls }
}

const ctx = { sessionId: "s1", workspaceDir: "/repo" }

test("notes are prepended to the prompt, for the workspace of the turn", async () => {
  const { handlers, calls } = host()
  const result = await handlers.get("before_prompt_build")!({ prompt: "history and request", currentUserMessage: "the request" }, ctx)
  assert.deepEqual(result, { prependContext: "note about tests" })
  assert.deepEqual(calls.find((c) => c[0] === "prompt")?.[1], { session_id: "s1", cwd: "/repo", prompt: "the request" })
})

test("an empty context changes nothing", async () => {
  const handlers = new Map<string, Function>()
  attach({ on: (name: string, handler: Function) => handlers.set(name, handler) }, async () => "")
  assert.equal(await handlers.get("before_prompt_build")!({ prompt: "x" }, ctx), undefined)
})

test("a failed tool call reports a non-zero exit", async () => {
  const seen: Array<Record<string, unknown>> = []
  const handlers = new Map<string, Function>()
  attach({ on: (name: string, handler: Function) => handlers.set(name, handler) }, async (event, payload) => {
    if (event === "tool") seen.push(payload)
    return ""
  })
  handlers.get("after_tool_call")!({ toolName: "exec", params: { command: "pytest" }, error: "exit 1" }, ctx)
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.deepEqual(seen[0].tool_response, { stderr: "exit 1", exit_code: 1 })
})

test("the entry registers every hook it needs", () => {
  const names: string[] = []
  const previous = process.env.KANERVA_COMMAND
  process.env.KANERVA_COMMAND = "kanerva"
  plugin.register({ on: (name: string) => names.push(name) })
  if (previous === undefined) delete process.env.KANERVA_COMMAND
  else process.env.KANERVA_COMMAND = previous
  assert.deepEqual(names.sort(), ["after_tool_call", "agent_end", "before_compaction", "before_prompt_build", "session_end"])
})
