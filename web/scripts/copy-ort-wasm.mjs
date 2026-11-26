import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");
const dstDir = path.join(rootDir, "public", "ort");

const candidates = [
  path.join(rootDir, "node_modules", "onnxruntime-web", "dist"),
  path.join(rootDir, "..", "packages", "lipsync-2d-web", "node_modules", "onnxruntime-web", "dist"),
];

const srcDir = candidates.find((dir) => fs.existsSync(dir));
if (srcDir) {
  fs.mkdirSync(dstDir, { recursive: true });
  for (const file of fs.readdirSync(srcDir)) {
    if (
      !file.endsWith(".map") &&
      (file === "ort-wasm-simd-threaded.mjs" ||
        file === "ort-wasm-simd-threaded.wasm" ||
        file === "ort-wasm-simd-threaded.jsep.mjs" ||
        file === "ort-wasm-simd-threaded.jsep.wasm")
    ) {
      fs.copyFileSync(path.join(srcDir, file), path.join(dstDir, file));
    }
  }
}
