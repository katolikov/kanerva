import assert from "node:assert/strict"
import { test } from "node:test"
import { attach, workspaceOf } from "./index.ts"

function host(context = "note about tests") {
  const handlers = new Map<string, Function>()
  const calls: Array<[string, Record<string, unknown>]> = []
  const amp = { system: { workspaceRoot: "file:///repo" }, on: (name: string, handler: Function) => handlers.set(name, handler) }
  attach(amp, async (event, payload) => {
    calls.push([event, payload])
    return event === "prompt" ? context : ""
  })
  return { handlers, calls }
}

const thread = { id: "T-1" }

test("the workspace comes from the file URI", () => {
  assert.equal(workspaceOf({ system: { workspaceRoot: "file:///repo/a" } }), "/repo/a")
  assert.equal(workspaceOf({ system: { workspaceRoot: null } }), process.cwd())
})

test("notes go in as a hidden message", async () => {
  const { handlers, calls } = host()
  const result = await handlers.get("agent.start")!({ thread, message: "how do I run the tests?" })
  assert.deepEqual(result, { message: { content: "note about tests", display: false } })
  assert.deepEqual(calls.find((c) => c[0] === "prompt")?.[1], { session_id: "T-1", cwd: "/repo", prompt: "how do I run the tests?" })
})

test("no notes means no message", async () => {
  const { handlers } = host("")
  assert.equal(await handlers.get("agent.start")!({ thread, message: "x" }), undefined)
})

test("failed tools report a non-zero exit, cancelled tools are skipped", async () => {
  const { handlers, calls } = host()
  handlers.get("tool.result")!({ thread, tool: "Bash", input: { cmd: "x" }, status: "cancelled" })
  handlers.get("tool.result")!({ thread, tool: "Bash", input: { command: "pytest" }, status: "error", error: "failed" })
  handlers.get("agent.end")!({ thread })
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.deepEqual(calls.map((c) => c[0]), ["tool", "stop"])
  assert.deepEqual(calls[0][1].tool_response, { stderr: "failed", exit_code: 1 })
})
