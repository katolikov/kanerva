import assert from "node:assert/strict"
import { test } from "node:test"
import { createPlugin, latestPrompt } from "./index.ts"

function host(context = "note about tests") {
  const calls: Array<[string, Record<string, unknown>]> = []
  const plugin = createPlugin(async (event, payload) => {
    calls.push([event, payload])
    return event === "prompt" ? context : ""
  })
  plugin.setup({}, { cwd: "/repo", session: { sessionId: "s1" } })
  return { plugin, calls }
}

const snapshot = { messages: [{ role: "user", content: [{ type: "text", text: '<user_input mode="yolo">how do I run the tests?</user_input>' }] }] }

test("the newest user text loses its wrapper", () => {
  assert.equal(latestPrompt(snapshot.messages), "how do I run the tests?")
  assert.equal(latestPrompt([{ role: "user", content: "older" }, { role: "assistant", content: "x" }]), "older")
  assert.equal(latestPrompt([]), "")
})

test("notes are appended at the start of a run", async () => {
  const { plugin, calls } = host()
  assert.deepEqual(await plugin.hooks.beforeRun({ snapshot }), { appendContext: "note about tests" })
  assert.deepEqual(calls.find((c) => c[0] === "prompt")?.[1], { session_id: "s1", cwd: "/repo", prompt: "how do I run the tests?" })
})

test("no notes means no extra context", async () => {
  const { plugin } = host("")
  assert.equal(await plugin.hooks.beforeRun({ snapshot }), undefined)
})

test("each command of a batch is recorded", async () => {
  const { plugin, calls } = host()
  plugin.hooks.afterTool({ toolCall: { name: "run_commands" }, input: { commands: ["pytest", "ls"] }, result: { isError: true } })
  plugin.hooks.afterRun()
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.deepEqual(calls.map((c) => c[0]), ["tool", "tool", "stop"])
  assert.deepEqual(calls[0][1].tool_input, { command: "pytest" })
  assert.deepEqual(calls[0][1].tool_response, { exit_code: 1 })
})
