import assert from "node:assert/strict"
import { test } from "node:test"
import { attach } from "./index.ts"

function host(context = "note about tests") {
  const handlers = new Map<string, Function>()
  const calls: Array<[string, Record<string, unknown>]> = []
  attach({ on: (name: string, handler: Function) => handlers.set(name, handler) }, async (event, payload) => {
    calls.push([event, payload])
    return event === "prompt" ? context : ""
  })
  return { handlers, calls }
}

const ctx = { cwd: "/repo", sessionManager: { getSessionId: () => "s1" } }

test("notes go in as a hidden message", async () => {
  const { handlers, calls } = host()
  const result = await handlers.get("before_agent_start")!({ prompt: "how do I run the tests?" }, ctx)
  assert.deepEqual(result, { message: { customType: "kanerva", content: "note about tests", display: false } })
  assert.deepEqual(calls.find((c) => c[0] === "prompt")?.[1], { session_id: "s1", cwd: "/repo", prompt: "how do I run the tests?" })
})

test("no notes means no message", async () => {
  const { handlers } = host("")
  assert.equal(await handlers.get("before_agent_start")!({ prompt: "x" }, ctx), undefined)
})

test("tool results, turn ends and compaction are recorded in order", async () => {
  const { handlers, calls } = host()
  handlers.get("tool_result")!({ toolName: "bash", input: { command: "pytest" }, isError: true }, ctx)
  handlers.get("agent_end")!({}, ctx)
  handlers.get("session_before_compact")!({}, ctx)
  handlers.get("session_compact")!({ compactionEntry: { summary: "we moved auth" } }, ctx)
  await handlers.get("session_shutdown")!({}, ctx)
  assert.deepEqual(calls.map((c) => c[0]), ["tool", "stop", "compact", "compact_summary"])
  assert.deepEqual(calls[0][1].tool_response, { exit_code: 1 })
})
