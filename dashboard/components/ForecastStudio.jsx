"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { TimeChart } from "@/components/charts";
import { fmt, compact, shortDate, weekday, addDays } from "@/lib/format";

const DAYS = 16;

export default function ForecastStudio({ stores, families, window: win, cutoff }) {
  const [store, setStore] = useState(44);
  const [family, setFamily] = useState("BEVERAGES");
  const [series, setSeries] = useState(null);       // exported history + forecast for the chosen store
  const [res, setRes] = useState(null);             // last API response
  const [mode, setMode] = useState("planned");      // planned | all | custom
  const [promoAll, setPromoAll] = useState(20);
  const [promoDays, setPromoDays] = useState(Array(DAYS).fill(0));
  const [hol, setHol] = useState(Array(DAYS).fill(false));
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const cache = useRef({});

  const dates = useMemo(() => Array.from({ length: DAYS }, (_, i) => addDays(win[0], i)), [win]);

  // load exported series for the store (history + exported forecast), cached per store
  useEffect(() => {
    let live = true;
    const get = cache.current[store]
      ? Promise.resolve(cache.current[store])
      : fetch(`/data/series/store_${store}.json`).then((r) => r.json()).then((j) => (cache.current[store] = j));
    get.then((j) => live && setSeries(j)).catch(() => live && setErr("Could not load this store's history."));
    return () => { live = false; };
  }, [store]);

  async function run(payload) {
    setBusy(true);
    setErr("");
    try {
      const r = await fetch("/api/forecast", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
      setRes(j);
      return j;
    } catch (e) {
      setErr(`The live forecast service did not answer (${e.message}). The chart shows the saved forecast; what-if changes need the service.`);
      setRes(null);
      return null;
    } finally {
      setBusy(false);
    }
  }

  // new store/family -> reset inputs to the real plan and fetch the baseline
  useEffect(() => {
    setMode("planned");
    run({ store, family }).then((j) => {
      if (j) {
        setPromoDays(j.inputs.promo_planned.map((v) => Math.round(v)));
        setHol(j.inputs.holiday_planned.map(Boolean));
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store, family]);

  function onRun() {
    const promo = mode === "planned" ? null : mode === "all" ? Array(DAYS).fill(Number(promoAll) || 0) : promoDays.map((v) => Number(v) || 0);
    const planned = res ? res.inputs.holiday_planned.map(Boolean) : null;
    const holChanged = planned ? hol.some((h, i) => h !== planned[i]) : hol.some(Boolean);
    run({ store, family, promo, holiday: holChanged ? hol.map((h) => (h ? 1 : 0)) : null });
  }

  const fam = series?.families?.[family];
  const histDays = 56;
  const chart = useMemo(() => {
    if (!fam) return [];
    const h = fam.history.slice(-histDays);
    const hStart = addDays(cutoff, -(h.length - 1));
    const rows = h.map((v, i) => ({ date: addDays(hStart, i), actual: v }));
    const base = res ? res.baseline : fam.forecast;
    const scen = res ? res.scenario : null;
    const changed = scen && scen.some((v, i) => Math.abs(v - base[i]) > 1e-6);
    dates.forEach((d, i) => {
      const s = changed ? scen[i] : base[i];
      const p90 = res ? res.scenario_p90[i] : s * (1 + fam.buffer90 / 100);
      rows.push({ date: d, baseline: base[i], scenario: changed ? scen[i] : undefined, band: [s, p90] });
    });
    const lastIdx = h.length - 1;
    rows[lastIdx].baseline = rows[lastIdx].actual;
    if (changed) rows[lastIdx].scenario = rows[lastIdx].actual;
    return { rows, changed };
  }, [fam, res, dates, cutoff]);

  const st = stores.find((s) => s.store === Number(store));
  const totals = res?.totals;
  const p90Total = res ? res.scenario_p90.reduce((a, b) => a + b, 0) : null;

  return (
    <>
      <section className="panel">
        <div className="controls">
          <div className="field">
            <label htmlFor="store">Store</label>
            <select id="store" value={store} onChange={(e) => setStore(Number(e.target.value))}>
              {stores.map((s) => <option key={s.store} value={s.store}>Store {s.store}, {s.city} (type {s.type})</option>)}
            </select>
          </div>
          <div className="field">
            <label htmlFor="family">Product family</label>
            <select id="family" value={family} onChange={(e) => setFamily(e.target.value)}>
              {families.map((f) => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>
          <div className="field">
            <span className="field-label">Items on promotion</span>
            <div className="seg" role="group" aria-label="Promotion input">
              {[["planned", "As planned"], ["all", "Same every day"], ["custom", "Day by day"]].map(([v, l]) => (
                <button key={v} type="button" aria-pressed={mode === v} onClick={() => setMode(v)}>{l}</button>
              ))}
            </div>
          </div>
          {mode === "all" && (
            <div className="field">
              <label htmlFor="promoAll">Items per day</label>
              <input id="promoAll" type="number" min="0" max="10000" value={promoAll} onChange={(e) => setPromoAll(e.target.value)} style={{ width: 110 }} />
            </div>
          )}
          <button className="btn" type="button" onClick={onRun} disabled={busy}>{busy ? "Forecasting…" : "Run forecast"}</button>
        </div>

        <div style={{ marginTop: 18 }}>
          <div className="field-label" style={{ marginBottom: 8 }}>
            Forecast days{mode === "custom" ? ": set promotion items per day" : ""}. Tick a day to treat it as a national holiday.
          </div>
          <div className="daygrid">
            {dates.map((d, i) => (
              <div className={`day${hol[i] ? " is-hol" : ""}`} key={d}>
                <div className="day-date">{weekday(d)} {shortDate(d)}</div>
                {mode === "custom" && (
                  <input type="number" min="0" max="10000" aria-label={`Promotion items on ${d}`} value={promoDays[i]}
                    onChange={(e) => setPromoDays(promoDays.map((v, j) => (j === i ? e.target.value : v)))} />
                )}
                {mode !== "custom" && (
                  <div className="small muted" style={{ marginTop: 4 }}>
                    Promo {mode === "all" ? promoAll : res ? Math.round(res.inputs.promo_planned[i]) : "–"}
                  </div>
                )}
                <label className="day-hol">
                  <input type="checkbox" checked={hol[i]} onChange={() => setHol(hol.map((h, j) => (j === i ? !h : h)))} />
                  Holiday
                </label>
              </div>
            ))}
          </div>
        </div>
      </section>

      {err && <div className="alert" style={{ marginTop: 16 }} role="status">{err}</div>}

      <section className="section panel">
        <div className="section-head">
          <div>
            <div className="chart-title">{family}, store {store}{st ? ` (${st.city})` : ""}</div>
            <div className="chart-note">Actual daily units for the last {histDays} days, then the 16-day forecast. The shaded band is the extra stock to cover 90% of days.</div>
          </div>
        </div>
        {fam ? (
          <TimeChart
            data={chart.rows}
            height={340}
            yFormat="compact"
            tipFormat="one"
            series={[
              { key: "actual", label: "Actual sales", color: "var(--s-actual)", width: 1.6 },
              { key: "baseline", label: "Forecast with planned inputs", color: "var(--s-forecast)", width: 2.4 },
              ...(chart.changed ? [{ key: "scenario", label: "Forecast with your inputs", color: "var(--s-scenario)", width: 2.4 }] : []),
            ]}
            band={{ key: "band", label: "Safety buffer to 90% service" }}
            marker={cutoff}
            markerLabel="Forecast made here"
          />
        ) : <p className="muted" style={{ padding: "40px 0" }}>Loading history…</p>}

        {res && (
          <div className="figures" style={{ marginTop: 18 }}>
            <div className="figure">
              <div className="figure-value num">{compact(totals.baseline)}</div>
              <div className="figure-label">Units in 16 days with the planned inputs</div>
            </div>
            <div className="figure">
              <div className="figure-value num">{compact(totals.scenario)}</div>
              <div className="figure-label">Units with your inputs</div>
              <div className={`figure-sub ${totals.change_pct > 0 ? "up" : totals.change_pct < 0 ? "down" : ""}`}>
                {totals.change_pct === 0 ? "No change" : `${totals.change_pct > 0 ? "+" : ""}${fmt(totals.change_pct, 1)}% vs planned`}
              </div>
            </div>
            <div className="figure">
              <div className="figure-value num">{compact(p90Total)}</div>
              <div className="figure-label">Order to cover 90% of days</div>
              <div className="figure-sub">Forecast plus {fmt(res.buffer_pct.p90, 0)}% buffer for this family</div>
            </div>
            <div className="figure">
              <div className="figure-value num">{res.never_sold ? "0" : `${fmt(res.buffer_pct.p80, 0)}%`}</div>
              <div className="figure-label">{res.never_sold ? "This store has never sold this family" : "Buffer for 80% of days"}</div>
              <div className="figure-sub">Learned from past windows, checked on the holdout</div>
            </div>
          </div>
        )}
      </section>

      {res && (
        <section className="section panel">
          <div className="chart-title">Day by day</div>
          <div className="table-wrap" style={{ marginTop: 10 }}>
            <table>
              <thead>
                <tr><th>Date</th><th>Day</th><th className="r">Promo planned</th><th className="r">Promo used</th><th>Holiday</th>
                  <th className="r">Forecast, planned</th><th className="r">Forecast, yours</th><th className="r">Order for 80%</th><th className="r">Order for 90%</th></tr>
              </thead>
              <tbody>
                {res.dates.map((d, i) => (
                  <tr key={d}>
                    <td>{shortDate(d)}</td><td>{weekday(d)}</td>
                    <td className="r">{fmt(res.inputs.promo_planned[i])}</td><td className="r">{fmt(res.inputs.promo_used[i])}</td>
                    <td>{res.inputs.holiday_used[i] ? "Yes" : ""}</td>
                    <td className="r">{fmt(res.baseline[i], 1)}</td><td className="r">{fmt(res.scenario[i], 1)}</td>
                    <td className="r">{fmt(res.scenario_p80[i], 1)}</td><td className="r">{fmt(res.scenario_p90[i], 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  );
}
