// Local static server for the harness page and the media files (Range support: decoders need it; cross-origin isolation headers: SharedArrayBuffer).
import fs from "node:fs"; import http from "node:http"; import path from "node:path"; import { fileURLToPath } from "node:url";
const here = path.dirname(fileURLToPath(import.meta.url));
export function makeServer(files) {
  return http.createServer((req, res) => {
  res.setHeader("cross-origin-opener-policy", "same-origin"); res.setHeader("cross-origin-embedder-policy", "require-corp"); res.setHeader("cross-origin-resource-policy", "cross-origin");   // SharedArrayBuffer (the encoder's audio sink) needs cross-origin isolation
  if (process.env.VERBOSE) console.error("[http]", req.method, req.url, req.headers.range || ""); const m = /^\/f\/(\d+)/.exec(req.url); if (req.url === "/" || req.url === "/index.html") { res.setHeader("content-type", "text/html"); return res.end('<!doctype html><body><script src="/harness.js"></script></body>'); }
  if (req.url === "/harness.js") { res.setHeader("content-type", "text/javascript"); return res.end(fs.readFileSync(path.join(here, "dist/harness.js"))); }
  const f = m && files[+m[1]]; if (!f) { res.statusCode = 404; return res.end(); }
  if (req.method === "HEAD") { const st = fs.statSync(f); res.setHeader("accept-ranges", "bytes"); res.setHeader("content-length", st.size); res.setHeader("content-type", { ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp" }[path.extname(f).toLowerCase()] || "application/octet-stream"); return res.end(); }
  const size = fs.statSync(f).size, r = /bytes=(\d*)-(\d*)/.exec(req.headers.range || ""); res.setHeader("accept-ranges", "bytes"); res.setHeader("access-control-allow-origin", "*");
  res.setHeader("content-type", { ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp" }[path.extname(f).toLowerCase()] || "application/octet-stream");
  if (r) { const a = r[1] ? +r[1] : 0, b = r[2] ? +r[2] : size - 1; res.statusCode = 206; res.setHeader("content-range", `bytes ${a}-${b}/${size}`); res.setHeader("content-length", b - a + 1); fs.createReadStream(f, { start: a, end: b }).pipe(res); }
  else { res.setHeader("content-length", size); const st = fs.createReadStream(f); if (process.env.VERBOSE) { let n = 0; st.on("data", (c) => (n += c.length)); res.on("close", () => console.error("[http] full GET closed after", n, "of", size, "finished:", res.writableFinished)); st.on("error", (e) => console.error("[http] stream error", e.message)); } st.pipe(res); }
});
}
