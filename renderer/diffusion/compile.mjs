// Compiles a composition (TSX using Pascal-case elements from "@diffusionstudio/jsx") to the CommonJS bundle `mount()` evaluates:
// babel-preset-solid (universal, moduleName @diffusionstudio/jsx) then esbuild ESM->CJS. Same recipe as the editor's own compile step.
import { transformSync } from "@babel/core"; import { transformSync as es } from "esbuild"; import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
export function compile(tsx, filename = "composition.tsx") {
  const r = transformSync(tsx, { filename, presets: [[require("babel-preset-solid"), { generate: "universal", moduleName: "@diffusionstudio/jsx" }], require("@babel/preset-typescript")], babelrc: false, configFile: false });
  return es(r.code, { format: "cjs", target: "es2022" }).code;
}
