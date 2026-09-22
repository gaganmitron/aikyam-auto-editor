// Bundles the harness (vendored runtime + encoder + reconciler) into dist/harness.js for headless Chromium.
import { build } from "esbuild"; import path from "node:path"; import { fileURLToPath } from "node:url";
const here = path.dirname(fileURLToPath(import.meta.url)), V = path.join(here, "vendor/editor"), P = (n) => path.join(V, "packages", n, "src/index.ts");
await build({ entryPoints: [path.join(here, "harness/entry.ts")], bundle: true, format: "iife", platform: "browser", target: "chrome140", outfile: path.join(here, "dist/harness.js"),
  alias: { "@diffusionstudio/runtime": P("runtime"), "@diffusionstudio/encoder": P("encoder"), "@diffusionstudio/reconciler": P("reconciler"), "@diffusionstudio/jsx": P("jsx"),
           "@diffusionstudio/assets": P("assets"), "@diffusionstudio/koota-solid": P("koota-solid") },
  nodePaths: [path.join(V, "node_modules")], conditions: ["browser", "solid", "import"], banner: { js: "\"use strict\";" }, logLevel: "info", minify: false, sourcemap: false });
