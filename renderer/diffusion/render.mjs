// node render.mjs composition.tsx out.mp4 [--fps 30] [--bitrate 8000000] [--timeout 1800]
// Renders a composition headlessly in Chromium through diffusionstudio/editor's runtime + encoder. Absolute file paths in quotes ("/x/y.mp4") are served over
// a local HTTP server (with Range support, decoders need it) and replaced by their URL. Picture only: audio is muxed afterwards by aikyam_video (WebCodecs here cannot encode AAC).
import fs from "node:fs"; import http from "node:http"; import path from "node:path"; import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core"; import { makeServer } from "./serve.mjs"; import { compile } from "./compile.mjs";
const here = path.dirname(fileURLToPath(import.meta.url));
const [src, out, ...rest] = process.argv.slice(2); const opt = (k, d) => { const i = rest.indexOf("--" + k); return i >= 0 ? rest[i + 1] : d; };
const fps = +opt("fps", 30), bitrate = +opt("bitrate", 8e6), timeoutS = +opt("timeout", 1800);
const files = []; let tsx = fs.readFileSync(src, "utf8");
const server = makeServer(files);
await new Promise((r) => server.listen(0, "127.0.0.1", r)); const base = `http://127.0.0.1:${server.address().port}`;
tsx = tsx.replace(/"(\/[^"\n]+\.(?:mp4|webm|mov|jpg|jpeg|png|webp))"/gi, (_, p) => { let i = files.indexOf(p); if (i < 0) { files.push(p); i = files.length - 1; } return `"${base}/f/${i}${path.extname(p)}"`; });
for (const f of files) if (!fs.existsSync(f)) throw new Error("missing asset " + f);
const code = compile(tsx, src);
// The profile lives on a real disk: Chromium spills blobs (fetch().blob() of a >14 MB clip, which the asset library does) to the profile dir, and a small /tmp tmpfs makes those fetches fail.
const profile = process.env.AIKYAM_CHROME_PROFILE || path.join(here, ".profile"); fs.mkdirSync(profile, { recursive: true });
const browser = await puppeteer.launch({ executablePath: process.env.CHROMIUM || "/usr/bin/chromium", headless: "new", userDataDir: profile, protocolTimeout: timeoutS * 1000,
  args: ["--no-sandbox", "--disable-gpu", "--autoplay-policy=no-user-gesture-required", "--enable-features=WebCodecs"] });
let ok = false; const chunks = [];
try {
  const page = await browser.newPage(); page.on("console", async (m) => { let t = m.text(); if (/JSHandle/.test(t)) t = t.replace(/JSHandle@error/g, "") + " " + (await Promise.all(m.args().map((a) => a.evaluate((e) => (e instanceof Error ? e.message + " | " + String(e.stack).split("\n").slice(0, 3).join(" > ") : String(e))).catch(() => "?")))).join(" ; "); if (process.env.VERBOSE || /error|fail/i.test(t)) console.error("[page]", t.slice(0, 300)); }); page.on("requestfailed", (r) => console.error("[requestfailed]", r.url().slice(-30), r.failure()?.errorText)); page.on("pageerror", (e) => console.error("[pageerror]", String(e).slice(0, 400)));
  if (process.env.VERBOSE) { const cdp = await page.createCDPSession(); await cdp.send("Network.enable"); cdp.on("Network.loadingFailed", (e) => console.error("[netfail]", e.errorText, e.blockedReason || "", JSON.stringify(e.corsErrorStatus || ""), e.canceled)); }
  await page.exposeFunction("__chunk", (b64) => { chunks.push(Buffer.from(b64, "base64")); });
  let last = -1; await page.exposeFunction("__progress", (p) => { const q = Math.floor(p * 10); if (q !== last) { last = q; process.stderr.write(`render ${q * 10}%\n`); } });
  await page.goto(base + "/"); await page.waitForFunction("window.__ready === true", { timeout: 30000 });
  const r = await page.evaluate((c, cfg) => window.__render(c, cfg), code, { fps, videoBitrate: bitrate }); fs.writeFileSync(out, Buffer.concat(chunks)); ok = true; console.log(JSON.stringify({ out, bytes: r.bytes }));
} finally { await browser.close(); server.close(); }
process.exit(ok ? 0 : 1);
