import Link from "next/link";
import PageHead from "@/components/PageHead";
import { TimeChart, BarsH } from "@/components/charts";
import { load } from "@/lib/data";
import { fmt, compact } from "@/lib/format";

export default function Overview() {
  const meta = load("meta.json");
  const fc = load("forecast.json");
  const k = meta.kpis;
  const last = fc.history[fc.history.length - 1];
  const data = [
    ...fc.history.map((d) => ({ date: d.date, actual: d.sales })),
    ...fc.forecast.map((d) => ({ date: d.date, forecast: d.sales })),
  ];
  const join = data.find((d) => d.date === last.date);
  join.forecast = last.sales; // connect the forecast line to the last known day
  const fam = fc.by_family.slice(0, 12).map((d) => ({ name: d.family, value: d.per_day }));

  return (
    <>
      <PageHead
        title="Store demand forecast"
        intro={`Daily unit sales for ${meta.n_series.toLocaleString()} store and product-family combinations of Corporación Favorita, a grocery chain in Ecuador, forecast ${meta.horizon} days ahead so stores can order the right amount.`}
        meta={meta}
      />

      <section className="section panel">
        <div className="section-head">
          <div>
            <div className="chart-title">Total units sold per day, all stores</div>
            <div className="chart-note">Last 4 months of history and the model&apos;s forecast for {meta.forecast_window[0]} to {meta.forecast_window[1]}: {compact(fc.total)} units in 16 days.</div>
          </div>
          <Link className="btn" href="/forecast" style={{ textDecoration: "none" }}>Open forecast studio</Link>
        </div>
        <TimeChart
          data={data}
          height={330}
          series={[
            { key: "actual", label: "Actual sales", color: "var(--s-actual)", width: 1.6 },
            { key: "forecast", label: "Forecast", color: "var(--s-forecast)", width: 2.4 },
          ]}
          shadeFrom={fc.forecast[0].date}
          shadeTo={fc.forecast[fc.forecast.length - 1].date}
          marker={last.date}
          markerLabel="Last known day"
        />
      </section>

      <section className="section">
        <div className="figures">
          <div className="figure">
            <div className="figure-value num">{fmt(k.holdout_wape, 1)}%</div>
            <div className="figure-label">Forecast error on an untouched 16-day test window (WAPE)</div>
            <div className="figure-sub">Seasonal-naive rule of thumb: {fmt(k.holdout_naive, 1)}%</div>
          </div>
          <div className="figure">
            <div className="figure-value num">{fmt(k.overstock_reduction, 0)}% less</div>
            <div className="figure-label">Over-ordered units than the rule of thumb</div>
            <div className="figure-sub">Six backtest windows, all stores</div>
          </div>
          <div className="figure">
            <div className="figure-value num">{fmt(k.understock_reduction, 0)}% less</div>
            <div className="figure-label">Units missing from the shelf (under-forecast)</div>
            <div className="figure-sub">Same six windows</div>
          </div>
          <div className="figure">
            <div className="figure-value num">{k.families_beating_naive}/{k.n_families}</div>
            <div className="figure-label">Product families where the model beats the rule of thumb</div>
            <div className="figure-sub">Stores: {k.stores_beating_naive}/{k.n_stores}{k.kaggle_score ? `. Kaggle RMSLE ${k.kaggle_score}` : ""}</div>
          </div>
        </div>
      </section>

      <section className="section grid-2">
        <div className="panel">
          <div className="chart-title">Forecast by product family</div>
          <div className="chart-note">Average units per day, 16 to 31 August 2017, 12 largest families</div>
          <BarsH data={fam} bars={[{ key: "value", label: "Units per day", color: "var(--s-forecast)" }]} xFormat="compact" labelFormat="compact" />
        </div>
        <div className="panel">
          <div className="chart-title">What the model knows on the forecast date</div>
          <div className="chart-note">{meta.features.length} inputs, all known 16 days ahead, so nothing from the future leaks in</div>
          <div className="table-wrap" style={{ marginTop: 12 }}>
            <table>
              <thead><tr><th>Input group</th><th>Inputs</th></tr></thead>
              <tbody>
                {Object.entries(meta.features.reduce((a, f) => ({ ...a, [f.group]: [...(a[f.group] || []), f.name] }), {})).map(([g, fs]) => (
                  <tr key={g}><td>{g}</td><td style={{ whiteSpace: "normal" }}>{fs.join(", ")}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note" style={{ marginTop: 12 }}>
            The final model is a tuned histogram gradient-boosting regressor trained on {fmt(meta.training_rows)} rows. <Link href="/model">How it works</Link>
          </p>
        </div>
      </section>
    </>
  );
}