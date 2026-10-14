import fs from "node:fs";
import path from "node:path";

// Server-side loader: JSON files written by src/10_export_dashboard.py (read at build time -> static pages)
export function load(name) {
  const p = path.join(process.cwd(), "public", "data", name);
  return JSON.parse(fs.readFileSync(p, "utf8"));
}
