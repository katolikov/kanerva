import assert from "node:assert/strict"
import { chmodSync, mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { test } from "node:test"
import { Bridge, cliRunner, promptText, resolveLaunch, summaryText } from "./core.ts"
import plugin, { serverV1, setupV2 } from "./index.ts"

const launch = { command: ["kanerva"], environment: {} }

function fake() {
  const calls: Array<[string, Record<string, unknown>]> = []
  const run = async (event: string, payload: Record<string, unknown>) => {
    calls.push([event, payload])
    return event === "prompt" ? "note about tests" : ""
  }
  return { calls, run, bridge: new Bridge(run, "/repo") }
}

function binDir(...names: string[]) {
  const dir = mkdtempSync(join(tmpdir(), "kanerva-bin-"))
  for (const name of names) {
    writeFileSync(join(dir, name), "#!/bin/sh\n")
    chmodSync(join(dir, name), 0o755)
  }
  return dir
}

test("a prompt's context is re-sent on each model call", async () => {
  const { bridge, calls } = fake()
  await bridge.prompt("s1", "how do I run the tests?")
  assert.equal(bridge.system("s1"), "note about tests")
  assert.equal(bridge.system("s1"), "note about tests")
  assert.equal(bridge.system("other"), "")
  assert.deepEqual(calls.find((c) => c[0] === "prompt"), ["prompt", { session_id: "s1", cwd: "/repo", prompt: "how do I run the tests?" }])
})

test("a resumed session starts on its first prompt, and only once", async () => {
  const { bridge, calls } = fake()
  await bridge.sessionStart("s1")
  await bridge.prompt("s1", "one")
  await bridge.prompt("s1", "two")
  await bridge.prompt("s2", "three")
  assert.equal(calls.filter((c) => c[0] === "session_start").length, 2)
})

test("writes run in order and do not block the caller", async () => {
  const order: string[] = []
  let release = () => {}
  const gate = new Promise<void>((resolve) => (release = resolve))
  const bridge = new Bridge(async (event) => {
    if (event === "tool") await gate
    order.push(event)
    return ""
  }, "/repo")
  bridge.toolAfter("s1", "bash", { command: "pytest" }, { stdout: "ok" })
  bridge.idle("s1")
  bridge.compacting("s1")
  bridge.compacted("s1", "")
  bridge.compacted("s1", "summary")
  assert.deepEqual(order, [])
  release()
  await bridge.flush()
  assert.deepEqual(order, ["tool", "stop", "compact", "compact_summary"])
})

test("a failing write does not stop later writes", async () => {
  const seen: string[] = []
  const bridge = new Bridge(async (event) => {
    seen.push(event)
    if (event === "tool") throw new Error("boom")
    return ""
  }, "/repo")
  bridge.toolAfter("s1", "bash", {}, {})
  bridge.idle("s1")
  await bridge.flush()
  assert.deepEqual(seen, ["tool", "stop"])
})

test("closing a session drops its state", async () => {
  const { bridge, calls } = fake()
  await bridge.prompt("s1", "hello there")
  bridge.close("s1")
  assert.equal(bridge.system("s1"), "")
  await bridge.sessionStart("s1")
  assert.equal(calls.filter((c) => c[0] === "session_start").length, 2)
})

test("prompt text keeps only text parts", () => {
  assert.equal(promptText([{ type: "text", text: "a" }, { type: "file" }, { type: "text", text: "b" }]), "a\nb")
  assert.equal(promptText(undefined), "")
})

test("the summary is the newest message flagged as one", () => {
  const messages = [
    { info: { summary: true }, parts: [{ type: "text", text: "old" }] },
    { info: {}, parts: [{ type: "text", text: "chat" }] },
    { info: { summary: true }, parts: [{ type: "text", text: "new" }] },
    { info: {}, parts: [{ type: "text", text: "more chat" }] },
  ]
  assert.equal(summaryText(messages), "new")
  assert.equal(summaryText([]), "")
  assert.equal(summaryText(undefined), "")
})

test("launch prefers the override, then the CLI, then uv, then python3", () => {
  const none = binDir()
  const lib = mkdtempSync(join(tmpdir(), "kanerva-lib-"))
  assert.deepEqual(resolveLaunch({ PATH: binDir("kanerva"), KANERVA_COMMAND: "uvx kanerva" }, lib).command, ["uvx", "kanerva"])
  assert.deepEqual(resolveLaunch({ PATH: binDir("kanerva", "uv") }, lib).command, ["kanerva"])
  const viaUv = resolveLaunch({ PATH: binDir("uv") }, lib)
  assert.equal(viaUv.command[0], "uv")
  assert.ok(viaUv.command.includes("numpy>=1.26"))
  assert.ok(!viaUv.command.includes("model2vec>=0.6"))
  assert.equal(viaUv.environment.PYTHONPATH, lib)
  assert.ok(resolveLaunch({ PATH: binDir("uv"), KANERVA_EMBEDDER: "static" }, lib).command.includes("model2vec>=0.6"))
  assert.deepEqual(resolveLaunch({ PATH: none, PYTHONPATH: "/x" }, lib), {
    command: ["python3", "-m", "kanerva"],
    environment: { PYTHONPATH: `${lib}:/x` },
  })
})

test("the runner passes the payload to the CLI and returns its output", async () => {
  const dir = binDir()
  const script = join(dir, "echo.sh")
  writeFileSync(script, '#!/bin/sh\ncat - >/dev/null\nprintf "%s" "$2"\n')
  chmodSync(script, 0o755)
  assert.equal(await cliRunner({ command: [script], environment: {} })("prompt", { prompt: "hi" }), "prompt")
})

test("the runner gives up on a failed, missing or stuck CLI", async () => {
  const dir = binDir()
  const failing = join(dir, "fail.sh")
  writeFileSync(failing, '#!/bin/sh\necho partial\nexit 3\n')
  chmodSync(failing, 0o755)
  const stuck = join(dir, "stuck.sh")
  writeFileSync(stuck, "#!/bin/sh\nexec sleep 30\n")
  chmodSync(stuck, 0o755)
  assert.equal(await cliRunner({ command: [failing], environment: {} })("prompt", {}), "")
  assert.equal(await cliRunner({ command: [join(dir, "missing")], environment: {} })("prompt", {}), "")
  const start = Date.now()
  assert.equal(await cliRunner({ command: [stuck], environment: {} }, 50)("prompt", {}), "")
  assert.ok(Date.now() - start < 5000)
})

test("the default export serves both plugin generations", async () => {
  assert.equal(plugin.id, "kanerva")
  assert.equal(typeof plugin.setup, "function")
  const hooks = await plugin.server({ directory: "/repo" })
  for (const name of ["config", "event", "chat.message", "experimental.chat.system.transform", "tool.execute.after", "dispose"]) {
    assert.equal(typeof (hooks as any)[name], "function")
  }
})

test("V1 adds the MCP server and keeps a user's own entry", async () => {
  const { run } = fake()
  const hooks = await serverV1({ directory: "/repo" }, run, launch)
  const fresh: any = {}
  await hooks.config(fresh)
  assert.deepEqual(fresh.mcp.kanerva.command, ["kanerva", "mcp"])
  assert.equal(fresh.mcp.kanerva.type, "local")
  const mine = { mcp: { kanerva: { type: "remote", url: "http://x" } } } as any
  await hooks.config(mine)
  assert.equal(mine.mcp.kanerva.type, "remote")
})

test("a V1 session flows through prompt, context, tools, compaction and deletion", async () => {
  const { run, calls } = fake()
  const client = {
    session: { messages: async () => ({ data: [{ info: { summary: true }, parts: [{ type: "text", text: "we use tox" }] }] }) },
  }
  const hooks = await serverV1({ client, directory: "/repo" }, run, launch)
  await hooks.event({ event: { type: "session.created", properties: { info: { id: "s1" } } } })
  await hooks["chat.message"]({ sessionID: "s1" }, { parts: [{ type: "text", text: "how do I test?" }] })
  const output = { system: [] as string[] }
  await hooks["experimental.chat.system.transform"]({ sessionID: "s1" }, output)
  assert.deepEqual(output.system, ["note about tests"])
  await hooks["tool.execute.after"]({ sessionID: "s1", tool: "bash", args: { command: "tox" } }, { output: "ok" })
  await hooks["experimental.session.compacting"]({ sessionID: "s1" })
  await hooks.event({ event: { type: "session.compacted", properties: { sessionID: "s1" } } })
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "s1" } } })
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "s1" } } } })
  await hooks.dispose()
  assert.deepEqual(calls.map((c) => c[0]).sort(), ["compact", "compact_summary", "prompt", "session_start", "stop", "tool"])
  assert.equal(calls.find((c) => c[0] === "compact_summary")?.[1].compact_summary, "we use tox")
  const after = { system: [] as string[] }
  await hooks["experimental.chat.system.transform"]({ sessionID: "s1" }, after)
  assert.deepEqual(after.system, [])
})

function v2Context() {
  const hooks: Record<string, (event: any) => any> = {}
  const servers: Record<string, any> = {}
  const ctx = {
    location: { directory: "/repo" },
    mcp: { transform: async (fn: any) => fn({ set: (name: string, config: any) => (servers[name] = config) }) },
    session: { hook: async (name: string, fn: any) => void (hooks[name] = fn) },
    tool: { hook: async (name: string, fn: any) => void (hooks[`tool.${name}`] = fn) },
    event: {
      subscribe: async function* () {
        yield { type: "session.created", data: { sessionID: "s1" } }
        yield { type: "session.compaction.ended", data: { sessionID: "s1", summary: "we use tox" } }
        yield { type: "session.idle", data: { sessionID: "s1" } }
      },
    },
  }
  return { ctx, hooks, servers }
}

test("V2 registers every hook and the MCP server", async () => {
  const { ctx, hooks, servers } = v2Context()
  const stop = await setupV2(ctx, fake().run, { command: ["uv", "run", "kanerva"], environment: { PYTHONPATH: "/lib" } })
  assert.deepEqual(Object.keys(hooks).sort(), ["compaction", "context", "prompt", "tool.execute.after"])
  assert.deepEqual(servers.kanerva, {
    type: "local",
    command: ["uv", "run", "kanerva", "mcp"],
    environment: { PYTHONPATH: "/lib" },
    codemode: false,
  })
  await stop()
})

test("a V2 session flows through prompt, context, tools and idle", async () => {
  const { ctx, hooks } = v2Context()
  const { run, calls } = fake()
  const stop = await setupV2(ctx, run, launch)
  await hooks.prompt({ sessionID: "s1", prompt: { text: "how do I deploy?" } })
  const request = { sessionID: "s1", system: [] as unknown[] }
  hooks.context(request)
  assert.deepEqual(request.system, [{ type: "text", text: "note about tests" }])
  await hooks["tool.execute.after"]({ sessionID: "s1", tool: "bash", input: { command: "ls" }, status: "completed", result: {} })
  await hooks["tool.execute.after"]({ sessionID: "s1", tool: "bash", input: {}, status: "error", error: {} })
  await new Promise((resolve) => setImmediate(resolve))
  await stop()
  const names = calls.map((c) => c[0])
  assert.ok(names.includes("session_start") && names.includes("stop"))
  assert.equal(names.filter((n) => n === "tool").length, 1)
  assert.equal(calls.find((c) => c[0] === "prompt")?.[1].prompt, "how do I deploy?")
  assert.equal(calls.find((c) => c[0] === "compact_summary")?.[1].compact_summary, "we use tox")
})
