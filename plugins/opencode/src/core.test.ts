import assert from "node:assert/strict"
import { test } from "node:test"
import { Bridge, promptText } from "./core.ts"
import plugin from "./index.ts"

function fake() {
  const calls: Array<[string, Record<string, unknown>]> = []
  const run = async (event: string, payload: Record<string, unknown>) => {
    calls.push([event, payload])
    return event === "prompt" ? "note about tests" : ""
  }
  return { calls, bridge: new Bridge(run, "/repo") }
}

test("a prompt's context is re-sent on each model call", async () => {
  const { bridge, calls } = fake()
  await bridge.prompt("s1", "how do I run the tests?")
  assert.equal(bridge.system("s1"), "note about tests")
  assert.equal(bridge.system("s1"), "note about tests")
  assert.equal(bridge.system("other"), "")
  assert.deepEqual(calls[0], ["prompt", { session_id: "s1", cwd: "/repo", prompt: "how do I run the tests?" }])
})

test("tool, idle and compaction events reach the memory layer", async () => {
  const { bridge, calls } = fake()
  await bridge.toolAfter("s1", "bash", { command: "pytest" }, { stdout: "ok" })
  await bridge.idle("s1")
  await bridge.compacting("s1")
  await bridge.compacted("s1", "")
  assert.deepEqual(calls.map((c) => c[0]), ["tool", "stop", "compact"])
})

test("prompt text keeps only text parts", () => {
  assert.equal(promptText([{ type: "text", text: "a" }, { type: "file" }, { type: "text", text: "b" }]), "a\nb")
})

test("the default export serves both plugin generations", async () => {
  assert.equal(plugin.id, "kanerva")
  assert.equal(typeof plugin.setup, "function")
  const hooks = await plugin.server({ directory: "/repo" })
  for (const name of ["event", "chat.message", "experimental.chat.system.transform", "tool.execute.after"]) {
    assert.equal(typeof (hooks as any)[name], "function")
  }
})

test("the V2 setup registers every hook it needs", async () => {
  const session: string[] = []
  const tools: string[] = []
  const ctx = {
    location: { directory: "/repo" },
    session: { hook: async (name: string) => void session.push(name) },
    tool: { hook: async (name: string) => void tools.push(name) },
    event: { subscribe: async function* () {} },
  }
  const stop = await plugin.setup(ctx)
  assert.deepEqual(session, ["prompt", "context", "compaction"])
  assert.deepEqual(tools, ["execute.after"])
  stop()
})
