// Ask the served page the questions scripts/check_web_port.py answered, and compare.
// Usage: python scripts/check_web_port.py && node scripts/check_web_port.mjs [port|url]
// A full URL checks a deployed site; a bare port checks a local server.
const PAGE = process.argv[2] || "8099";
const t = (await (await fetch("http://127.0.0.1:9222/json")).json())
  .find((x) => x.type === "page" && x.url.includes(PAGE));
if (!t) { console.error(`No page matching ${PAGE} on the debug port`); process.exit(1); }

const ws = new WebSocket(t.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const pending = new Map();
ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } };
const send = (method, params = {}) => { const i = ++id; ws.send(JSON.stringify({ id: i, method, params })); return new Promise((r) => pending.set(i, r)); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function ev(expr) {
  const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || "eval failed");
  return r.result.result.value;
}
await send("Runtime.enable");
for (let i = 0; i < 120; i++) { if (await ev(`!!window.ort && !!document.querySelector('.cell')`)) break; await sleep(500); }

// The reference is scaffolding, not part of the site, so it is read off disk rather than
// fetched — a deployed target does not serve it.
const ref = JSON.parse(await (await import("node:fs/promises")).readFile("docs/data/.port_check.json", "utf8"));
const P = ref.patterns, C = ref.cases;
let checked = 0, failed = 0;
const fail = (what, want, got) => { failed++; console.log(`  FAIL ${what}\n    python: ${JSON.stringify(want)}\n    js    : ${JSON.stringify(got)}`); };

for (const [name, want] of Object.entries(C.tokenize)) {
  const got = await ev(`JSON.stringify(tokenize(${JSON.stringify(P[name])}))`);
  checked++;
  if (JSON.stringify(want) !== got) fail(`tokenize ${name}`, want, JSON.parse(got));
}

for (const [key, want] of Object.entries(C.paired)) {
  const [a, b] = key.split("|");
  const got = JSON.parse(await ev(`(function(){const d=describePair(${JSON.stringify(P[a])},${JSON.stringify(P[b])});return JSON.stringify([d.onset_balance,d.antiphony]);})()`));
  checked++;
  const close = (x, y) => (x === null || y === null) ? x === y : Math.abs(x - y) < 1e-6;
  if (!close(want[0], got[0]) || !close(want[1], got[1])) fail(`paired ${key}`, want, got);
}

for (const [ix, c] of C.key_shift.entries()) {
  const got = await ev(`keyShift(${JSON.stringify(c.heard)}, ${JSON.stringify(c.seg)})`);
  checked++;
  if (got !== c.shift) fail(`key_shift case ${ix}`, c.shift, got);
}

await ev(`document.getElementById('sampler').value = 'greedy'`);
// Keys are "<model id>|<pattern>": each model has its own weights, so each is asked separately
let current = null;
for (const [key, want] of Object.entries(C.greedy)) {
  const [model, name] = key.split("|");
  if (model !== current) {
    const { part_1, part_2 } = ref.models.find((m) => m.id === model);
    await ev(`(() => { const s = document.getElementById('inPart'); s.value = ${JSON.stringify(part_1)}; s.onchange(); })()`);
    for (let i = 0; i < 600; i++) {
      if (await ev(`!document.getElementById('inPart').disabled`)) break;
      await sleep(200);
    }
    await ev(`(() => { const s = document.getElementById('outPart'); s.value = ${JSON.stringify(part_2)}; s.onchange(); })()`);
    for (let i = 0; i < 600; i++) {
      if (await ev(`!document.getElementById('inPart').disabled`)) break;
      await sleep(200);
    }
    const got = await ev(`META.run`);
    if (got !== model) fail(`selecting ${part_1} -> ${part_2}`, model, got);
    current = model;

    // The score reads this model's exported weights out of META, so it can only be asked while
    // this model is the selected one
    const keys = Object.keys(C.interlock).filter((k) => k.startsWith(`${model}|`));
    const pairs = keys.map((k) => { const [, a, b] = k.split("|"); return [P[a], P[b]]; });
    const scores = JSON.parse(await ev(`JSON.stringify(${JSON.stringify(pairs)}.map(([a, b]) => interlockScore(a, b)))`));
    keys.forEach((k, i) => {
      checked++;
      if (Math.abs(scores[i] - C.interlock[k]) > 1e-6) fail(`interlock ${k}`, C.interlock[k], scores[i]);
    });
  }
  const got = JSON.parse(await ev(`generateBatch(${JSON.stringify(P[name])}, 1, 'greedy', 1.0, 0.92).then(r => JSON.stringify(r[0]))`));
  checked++;
  if (JSON.stringify(want) !== JSON.stringify(got)) fail(`greedy ${key}`, want, got);
}

console.log(`${checked} checks, ${failed} failed`);
ws.close();
process.exit(failed ? 1 : 0);
