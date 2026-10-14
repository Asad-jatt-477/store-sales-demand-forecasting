export const fmt = (v, d = 0) =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
export const pct = (v, d = 1) => (v === null || v === undefined ? "–" : `${fmt(v, d)}%`);
export const compact = (v) => {
  if (v === null || v === undefined) return "–";
  const a = Math.abs(v);
  if (a >= 1e6) return `${fmt(v / 1e6, 1)}M`;
  if (a >= 1e3) return `${fmt(v / 1e3, a >= 1e4 ? 0 : 1)}k`;
  return fmt(v, a < 10 ? 1 : 0);
};
// All dates are calendar days: compute in UTC so the viewer's time zone never shifts them
export const shortDate = (s) => new Date(`${s}T00:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
export const weekday = (s) => new Date(`${s}T00:00:00Z`).toLocaleDateString("en-US", { weekday: "short", timeZone: "UTC" });
export const addDays = (s, n) => {
  const d = new Date(`${s}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
};
export const windowLabel = (w) => w;
