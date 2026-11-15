import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const tmpDir = resolve(".tmp");

mkdirSync(tmpDir, { recursive: true });
process.env.TMPDIR = tmpDir;
process.env.TEMP = tmpDir;
process.env.TMP = tmpDir;

await import("../node_modules/vitest/vitest.mjs");
