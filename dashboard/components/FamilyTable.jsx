"use client";
import { useMemo, useState } from "react";
import { fmt } from "@/lib/format";

const COLS = [
  ["family", "Family", false],
  ["units", "Units (backtests)", true],
  ["wape", "Model WAPE", true],
  ["wape_naive", "Naive WAPE", true],
  ["bias", "Fcst ÷ actual", true],
  ["buffer80", "Buffer 80%", true],
  ["buffer90", "Buffer 90%", true],
  ["coverage90", "Holdout coverage at 90%", true],
];

export default function FamilyTable({ rows }) {
  const [sort, setSort] = useState(["units", -1]);
  const [q, setQ] = useState("");
  const data = useMemo(() => {
    const [k, d] = sort;
    return rows.filter((r) => r.family.toLowerCase().includes(q.toLowerCase()))
      .sort((a, b) => (typeof a[k] === "string" ? a[k].localeCompare(b[k]) : (a[k] ?? -1) - (b[k] ?? -1)) * d);
  }, [rows, sort, q]);
  const cell = (r, k) => {
    const v = r[k];
    if (v == null) return "–";
    if (k === "units") return fmt(v, 0);
    if (k === "bias") return fmt(v, 2);
    if (k.startsWith("buffer")) return `+${fmt(v, 0)}%`;
    return `${fmt(v, 1)}%`;
  };
  return (
    <>
      <div className="controls" style={{ marginTop: 14 }}>
        <div className="field">
          <label htmlFor="famq">Find a family</label>
          <input id="famq" type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. dairy" />
        </div>
      </div>
      <div className="table-wrap" style={{ marginTop: 12, maxHeight: 520 }}>
        <table className="sticky">
          <thead>
            <tr>{COLS.map(([k, l, num]) => (
              <th key={k} className={num ? "r" : ""} aria-sort={sort[0] === k ? (sort[1] > 0 ? "ascending" : "descending") : "none"}>
                <button type="button" className="th-sort" onClick={() => setSort([k, sort[0] === k ? -sort[1] : num ? -1 : 1])}>
                  {l}{sort[0] === k ? (sort[1] > 0 ? " ▲" : " ▼") : ""}
                </button>
              </th>
            ))}</tr>
          </thead>
          <tbody>{data.map((r) => (
            <tr key={r.family}>
              {COLS.map(([k, , num]) => (
                <td key={k} className={num ? "r" : ""}>
                  {k === "family" ? <>{r.family}{r.perishable && <span className="tag">perishable</span>}</> : cell(r, k)}
                  {k === "wape" && r.wape < r.wape_naive && <span className="sr-only"> (beats naive)</span>}
                </td>
              ))}
            </tr>
          ))}</tbody>
        </table>
      </div>
    </>
  );
}
