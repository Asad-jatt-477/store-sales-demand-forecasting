"use client";
import {
  ResponsiveContainer, ComposedChart, LineChart, Line, Area, Bar, BarChart, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ReferenceArea, Cell, LabelList,
} from "recharts";
import { compact, fmt, shortDate } from "@/lib/format";

// Server pages pass formatter NAMES (functions can't cross the server/client boundary)
const F = {
  compact: (v) => compact(v),
  int: (v) => fmt(v, 0),
  one: (v) => fmt(v, 1),
  two: (v) => fmt(v, 2),
  pct: (v) => `${fmt(v, 1)}%`,
  pp: (v) => `${v > 0 ? "+" : ""}${fmt(v, 1)}`,
  idx: (v) => fmt(v, 0),
  ratio: (v) => fmt(v, 3),
  date: (v) => shortDate(v),
  month: (v) => new Date(`${v}T00:00:00Z`).toLocaleDateString("en-US", { month: "short", year: "2-digit", timeZone: "UTC" }),
  raw: (v) => v,
};
const f = (name) => F[name] || F.raw;
const axis = { stroke: "var(--rule-strong)", tick: { fill: "var(--ink-3)", fontSize: 12 }, tickLine: false };

function TT({ active, payload, label, xFormat, yFormat, series }) {
  if (!active || !payload || !payload.length) return null;
  const rows = payload.filter((p) => p.value !== null && p.value !== undefined && !Array.isArray(p.value));
  const bands = payload.filter((p) => Array.isArray(p.value));
  return (
    <div className="tt">
      <div className="tt-title">{f(xFormat)(label)}</div>
      {rows.map((p) => {
        const s = (series || []).find((x) => x.key === p.dataKey) || {};
        return (
          <div className="tt-row" key={p.dataKey}>
            <span><i className="swatch" style={{ background: s.color || p.color }} />{s.label || p.name}</span>
            <b className="num">{f(yFormat)(p.value)}</b>
          </div>
        );
      })}
      {bands.map((p) => (
        <div className="tt-row" key={p.dataKey || "band"}>
          <span><i className="swatch area" style={{ background: "var(--band)" }} />{p.name}</span>
          <b className="num">{f(yFormat)(p.value[0])} to {f(yFormat)(p.value[1])}</b>
        </div>
      ))}
    </div>
  );
}

export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((s) => (
        <span key={s.label}>
          <i className={`swatch${s.dash ? " dash" : ""}${s.area ? " area" : ""}`} style={{ background: s.area ? "var(--band)" : s.color }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}

/** Time series: lines + optional range band + optional shaded region and marker line */
export function TimeChart({ data, x = "date", series, band, height = 300, xFormat = "date", yFormat = "compact", tipFormat,
  shadeFrom, shadeTo, marker, markerLabel, yDomain, legend = true, xTicks }) {
  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey={x} {...axis} tickFormatter={f(xFormat)} minTickGap={28} ticks={xTicks} />
          <YAxis {...axis} axisLine={false} tickFormatter={f(yFormat)} width={52} domain={yDomain || [0, "auto"]} />
          {shadeFrom && <ReferenceArea x1={shadeFrom} x2={shadeTo} fill="var(--brand-wash)" fillOpacity={0.6} ifOverflow="extendDomain" />}
          {marker && <ReferenceLine x={marker} stroke="var(--rule-strong)" strokeDasharray="3 3"
            label={markerLabel ? { value: markerLabel, position: "insideTopLeft", fill: "var(--ink-3)", fontSize: 11.5 } : undefined} />}
          {band && <Area dataKey={band.key} name={band.label} stroke="none" fill="var(--band)" fillOpacity={1} isAnimationActive={false} activeDot={false} />}
          {series.map((s) => (
            <Line key={s.key} dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={s.width || 2}
              strokeDasharray={s.dash ? "5 4" : undefined} dot={s.dots ? { r: 2.5, strokeWidth: 0, fill: s.color } : false}
              activeDot={{ r: 4, stroke: "var(--panel)", strokeWidth: 2 }} isAnimationActive={false} connectNulls={false} />
          ))}
          <Tooltip content={<TT xFormat={xFormat} yFormat={tipFormat || yFormat} series={series} />} cursor={{ stroke: "var(--ink-3)", strokeWidth: 1 }} />
        </ComposedChart>
      </ResponsiveContainer>
      {legend && series.length + (band ? 1 : 0) > 1 && (
        <Legend items={[...series.map((s) => ({ label: s.label, color: s.color, dash: s.dash })), ...(band ? [{ label: band.label, area: true }] : [])]} />
      )}
    </div>
  );
}

/** Horizontal bars (category on y). Optional second series drawn as a thin comparison bar. */
export function BarsH({ data, y = "name", bars, height, xFormat = "one", labelFormat, legend = true, highlight, refX }) {
  const h = height || Math.max(160, data.length * (bars.length > 1 ? 30 : 24) + 40);
  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height={h}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 48, bottom: 0, left: 4 }} barGap={2} barCategoryGap={bars.length > 1 ? "22%" : "28%"}>
          <CartesianGrid horizontal={false} stroke="var(--grid)" />
          <XAxis type="number" {...axis} tickFormatter={f(xFormat)} />
          <YAxis type="category" dataKey={y} {...axis} axisLine={false} width={170} interval={0} tick={{ fill: "var(--ink-2)", fontSize: 12 }} />
          {refX !== undefined && <ReferenceLine x={refX} stroke="var(--ink-3)" />}
          {bars.map((b) => (
            <Bar key={b.key} dataKey={b.key} name={b.label} fill={b.color} radius={[0, 4, 4, 0]} isAnimationActive={false} maxBarSize={18}>
              {highlight && b.key === bars[0].key && data.map((d, i) => (
                <Cell key={i} fill={d[highlight] ? b.color : "var(--rule-strong)"} />
              ))}
              {labelFormat && b.key === bars[0].key && (
                <LabelList dataKey={b.key} position="right" formatter={f(labelFormat)} style={{ fill: "var(--ink-2)", fontSize: 11.5 }} />
              )}
            </Bar>
          ))}
          <Tooltip content={<TT xFormat="raw" yFormat={xFormat} series={bars} />} cursor={{ fill: "var(--page)" }} />
        </BarChart>
      </ResponsiveContainer>
      {legend && bars.length > 1 && <Legend items={bars.map((b) => ({ label: b.label, color: b.color }))} />}
    </div>
  );
}

/** Vertical bars (category on x) */
export function BarsV({ data, x = "name", bars, height = 240, yFormat = "one", labelFormat, legend = true, refY, yDomain }) {
  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 18, right: 8, bottom: 0, left: 0 }} barGap={2}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey={x} {...axis} interval={0} />
          <YAxis {...axis} axisLine={false} tickFormatter={f(yFormat)} width={46} domain={yDomain || [0, "auto"]} />
          {refY !== undefined && <ReferenceLine y={refY} stroke="var(--ink-3)" strokeDasharray="3 3" />}
          {bars.map((b) => (
            <Bar key={b.key} dataKey={b.key} name={b.label} fill={b.color} radius={[4, 4, 0, 0]} isAnimationActive={false} maxBarSize={36}>
              {labelFormat && <LabelList dataKey={b.key} position="top" formatter={f(labelFormat)} style={{ fill: "var(--ink-2)", fontSize: 11 }} />}
            </Bar>
          ))}
          <Tooltip content={<TT xFormat="raw" yFormat={yFormat} series={bars} />} cursor={{ fill: "var(--page)" }} />
        </BarChart>
      </ResponsiveContainer>
      {legend && bars.length > 1 && <Legend items={bars.map((b) => ({ label: b.label, color: b.color }))} />}
    </div>
  );
}

/** Simple multi-line (non-date x), e.g. horizon day, weeks, trials */
export function Lines({ data, x, series, height = 240, xFormat = "raw", yFormat = "one", xLabel, yDomain }) {
  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: xLabel ? 18 : 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey={x} {...axis} tickFormatter={f(xFormat)}
            label={xLabel ? { value: xLabel, position: "insideBottom", offset: -12, fill: "var(--ink-3)", fontSize: 12 } : undefined} />
          <YAxis {...axis} axisLine={false} tickFormatter={f(yFormat)} width={46} domain={yDomain || ["auto", "auto"]} />
          {series.map((s) => (
            <Line key={s.key} dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={s.width || 2}
              strokeDasharray={s.dash ? "5 4" : undefined} dot={s.dots === false ? false : { r: 2.5, strokeWidth: 0, fill: s.color }}
              activeDot={{ r: 4, stroke: "var(--panel)", strokeWidth: 2 }} isAnimationActive={false} connectNulls />
          ))}
          <Tooltip content={<TT xFormat={xFormat} yFormat={yFormat} series={series} />} cursor={{ stroke: "var(--ink-3)", strokeWidth: 1 }} />
        </LineChart>
      </ResponsiveContainer>
      {series.length > 1 && <Legend items={series.map((s) => ({ label: s.label, color: s.color, dash: s.dash }))} />}
    </div>
  );
}
