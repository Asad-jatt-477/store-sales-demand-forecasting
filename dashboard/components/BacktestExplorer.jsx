"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { TimeChart } from "@/components/charts";
import { fmt, compact, addDays } from "@/lib/format";

const wape = (y, p) => {
  const s = y.reduce((a, b) => a + b, 0);
  return s > 0 ? (y.reduce((a, v, i) => a + Math.abs(v - p[i]), 0) / s) * 100 : null;
};

export default function BacktestExplorer({ stores, families, windows }) {
  const [win, setWin] = useState(windows[windows.length - 1].window);
  const [store, setStore] = useState(44);
  const [family, setFamily] = useState("BEVERAGES");
  const [data, setData] = useState(null);
  const cache = useRef({});

  useEffect(() => {
    let live = true;
    const get = cache.current[store]
      ? Promise.resolve(cache.current[store])
      : fetch(`/data/series/store_${store}.json`).then((r) => r.json()).then((j) => (cache.current[store] = j));
    get.then((j) => live && setData(j));
    return () => { live = false; };
  }, [store]);

  const w = windows.find((x) => x.window === win);
  const bt = data?.families?.[family]?.backtests?.[win];
  const rows = useMemo(() => (bt ? bt.actual.map((a, i) => ({ date: addDays(w.start, i), actual: a, model: bt.model[i], naive: bt.naive[i] })) : []), [bt, w]);
  const sum = (a) => a.reduce((x, y) => x + y, 0);

  return (
    <>
      <section className="panel">
        <div className="controls">
          <div className="field">
            <label htmlFor="win">Backtest window</label>
            <select id="win" value={win} onChange={(e) => setWin(e.target.value)}>
              {windows.map((x) => <option key={x.window} value={x.window}>{x.window}: {x.role}</option>)}
            </select>
          </div>
          <div className="field">
            <label htmlFor="bstore">Store</label>
            <select id="bstore" value={store} onChange={(e) => setStore(Number(e.target.value))}>
              {stores.map((s) => <option key={s.store} value={s.store}>Store {s.store}, {s.city}</option>)}
            </select>
          </div>
          <div className="field">
            <label htmlFor="bfam">Product family</label>
            <select id="bfam" value={family} onChange={(e) => setFamily(e.target.value)}>
              {families.map((f) => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>
        </div>
      </section>

      <section className="section panel">
        <div className="chart-title">{family}, store {store}: {win}</div>
        <div className="chart-note">
          Model trained only on data up to {w.cutoff}, then asked to forecast the next 16 days. The seasonal-naive line repeats
          the average of the same weekday over the previous weeks.
        </div>
        {bt ? (
          <>
            <TimeChart data={rows} height={300} yFormat="compact" tipFormat="one" series={[
              { key: "actual", label: "Actual sales", color: "var(--s-actual)", width: 2, dots: true },
              { key: "model", label: "Model forecast", color: "var(--s-forecast)", width: 2.2 },
              { key: "naive", label: "Seasonal naive", color: "var(--s-naive)", width: 1.6, dash: true },
            ]} />
            <div className="figures" style={{ marginTop: 16 }}>
              <div className="figure"><div className="figure-value">{compact(sum(bt.actual))}</div><div className="figure-label">Actual units, 16 days</div></div>
              <div className="figure"><div className="figure-value">{compact(sum(bt.model))}</div><div className="figure-label">Model forecast</div>
                <div className="figure-sub">{sum(bt.actual) > 0 ? `${fmt((sum(bt.model) / sum(bt.actual)) * 100, 0)}% of actual` : ""}</div></div>
              <div className="figure"><div className="figure-value">{wape(bt.actual, bt.model) === null ? "–" : `${fmt(wape(bt.actual, bt.model), 1)}%`}</div>
                <div className="figure-label">Model error for this series (WAPE)</div></div>
              <div className="figure"><div className="figure-value">{wape(bt.actual, bt.naive) === null ? "–" : `${fmt(wape(bt.actual, bt.naive), 1)}%`}</div>
                <div className="figure-label">Seasonal-naive error</div></div>
            </div>
            <p className="note" style={{ marginTop: 12 }}>
              A single store and family is noisy: some series beat the rule of thumb by a lot, a few lose. The window totals below are the fair comparison.
            </p>
          </>
        ) : <p className="muted" style={{ padding: "40px 0" }}>Loading…</p>}
      </section>
    </>
  );
}
