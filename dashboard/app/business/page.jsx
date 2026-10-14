import PageHead from "@/components/PageHead";
import { BarsH, BarsV } from "@/components/charts";
import FamilyTable from "@/components/FamilyTable";
import { load } from "@/lib/data";
import { fmt, compact } from "@/lib/format";

export const metadata = { title: "Business impact | Favorita Demand Forecast" };

export default function Business() {
  const meta = load("meta.json");
  const b = load("business.json");
  const ev = load("evaluation.json");
  const all = b.business.find((x) => x.group === "All families") || b.business[0];
  const stock = b.business.map((g) => ({
    name: g.group,
    over_m: g.overstock_model, over_n: g.overstock_naive, under_m: g.understock_model, under_n: g.understock_naive,
  }));
  const fams = ev.families.map((f) => {
    const buf = b.buffers.find((x) => x.family === f.family) || {};
    return { ...f, ...buf, perishable: b.perishables.includes(f.family) };
  });
  const errShare = [...ev.families].sort((a, c) => c.error_share - a.error_share).slice(0, 10)
    .map((f) => ({ name: f.family, value: f.error_share }));
  const storesWorst = [...ev.stores].sort((a, c) => c.wape - a.wape).slice(0, 10)
    .map((s) => ({ name: `Store ${s.store}, ${s.city}`, model: s.wape, naive: s.wape_naive }));
  const types = ev.store_types.map((t) => ({ name: `Type ${t.type}`, model: t.wape, naive: t.wape_naive }));
  const days = [...ev.day_kind, ...ev.promo].map((d) => ({ name: d.kind, model: d.wape, naive: d.wape_naive }));
  const growth = ev.growth_vs_bias.map((g) => ({ name: g.window.replace("Validation (Aug 2017)", "Validation").replace(/^Holdout.*/, "Holdout"), bias: g.bias, growth: g.growth_yoy }));
  const cov90 = b.buffers.reduce((a, x) => a + x.coverage90 * x.units_share, 0) / b.buffers.reduce((a, x) => a + x.units_share, 0);

  return (
    <>
      <PageHead
        title="Business impact"
        intro="Forecast accuracy only matters if it changes what lands on the shelf. These numbers translate the backtest errors into stock: units ordered that did not sell, and demand that found an empty shelf."
        meta={meta}
      />

      <section className="figures section">
        <div className="figure"><div className="figure-value">−{fmt(all.overstock_cut_pct, 0)}%</div><div className="figure-label">Excess stock vs the seasonal rule of thumb</div>
          <div className="figure-sub">{compact(all.overstock_naive - all.overstock_model)} fewer units over-ordered</div></div>
        <div className="figure"><div className="figure-value">−{fmt(all.understock_cut_pct, 0)}%</div><div className="figure-label">Missed demand vs the rule of thumb</div>
          <div className="figure-sub">{compact(all.understock_naive - all.understock_model)} more units on the shelf when needed</div></div>
        <div className="figure"><div className="figure-value">{meta.kpis.families_beating_naive}/{meta.kpis.n_families}</div><div className="figure-label">Families where the model wins</div></div>
        <div className="figure"><div className="figure-value">{meta.kpis.stores_beating_naive}/{meta.kpis.n_stores}</div><div className="figure-label">Stores where the model wins</div></div>
      </section>

      <section className="section grid-2">
        <div className="panel">
          <div className="chart-title">Excess stock, units over all 6 backtests</div>
          <div className="chart-note">Units forecast but not sold: money tied up, and for perishables, waste.</div>
          <BarsV data={stock} yFormat="compact" labelFormat="compact" bars={[
            { key: "over_m", label: "Model", color: "var(--s-forecast)" },
            { key: "over_n", label: "Seasonal naive", color: "var(--s-naive)" },
          ]} />
        </div>
        <div className="panel">
          <div className="chart-title">Missed demand, units over all 6 backtests</div>
          <div className="chart-note">Units sold above the forecast: if stock followed the forecast, these risked an empty shelf.</div>
          <BarsV data={stock} yFormat="compact" labelFormat="compact" bars={[
            { key: "under_m", label: "Model", color: "var(--s-forecast)" },
            { key: "under_n", label: "Seasonal naive", color: "var(--s-naive)" },
          ]} />
        </div>
      </section>
      <p className="note">Perishable families: {b.perishables.map((x) => x.toLowerCase()).join(", ")}. Their reduction is smaller ({fmt(b.business.find((x) => x.group === "Perishable")?.overstock_cut_pct, 0)}% vs {fmt(b.business.find((x) => x.group === "Non-perishable")?.overstock_cut_pct, 0)}% less excess stock), but it is still a clear gain.</p>

      <h2 className="section-title">Safety stock</h2>
      <section className="panel">
        <div className="chart-title">How much extra to order, per family</div>
        <div className="chart-note">
          The buffer is learned from past backtest errors: forecast plus this % covers demand on 80% or 90% of days. It was then checked on the untouched holdout,
          where the 90% buffer covered {fmt(cov90, 1)}% of days on average, weighting each family by its volume.
        </div>
        <FamilyTable rows={fams} />
      </section>

      <h2 className="section-title">Where the remaining error sits</h2>
      <section className="grid-2">
        <div className="panel">
          <div className="chart-title">Share of total error by family</div>
          <div className="chart-note">Big families dominate the error in units; improving them pays most.</div>
          <BarsH data={errShare} xFormat="pct" labelFormat="pct" bars={[{ key: "value", label: "Share of error", color: "var(--s-forecast)" }]} />
        </div>
        <div className="panel">
          <div className="chart-title">The 10 hardest stores</div>
          <div className="chart-note">WAPE %. Even in its hardest stores the model is better than the rule of thumb.</div>
          <BarsH data={storesWorst} xFormat="one" labelFormat="one" bars={[
            { key: "model", label: "Model", color: "var(--s-forecast)" },
            { key: "naive", label: "Seasonal naive", color: "var(--s-naive)" },
          ]} />
        </div>
      </section>
      <section className="section grid-2">
        <div className="panel">
          <div className="chart-title">By store type</div>
          <div className="chart-note">WAPE %</div>
          <BarsV data={types} yFormat="one" labelFormat="one" bars={[
            { key: "model", label: "Model", color: "var(--s-forecast)" },
            { key: "naive", label: "Seasonal naive", color: "var(--s-naive)" },
          ]} />
        </div>
        <div className="panel">
          <div className="chart-title">Holidays and promotions</div>
          <div className="chart-note">WAPE %. Local holidays are hardest: the model only knows national ones.</div>
          <BarsH data={days} xFormat="one" labelFormat="one" bars={[
            { key: "model", label: "Model", color: "var(--s-forecast)" },
            { key: "naive", label: "Seasonal naive", color: "var(--s-naive)" },
          ]} />
        </div>
      </section>

      <section className="section panel">
        <div className="chart-title">Over- or under-forecast, and why</div>
        <div className="chart-note">
          Forecast ÷ actual per window, next to year-on-year growth. When the business grows fast the model lags slightly behind; in 2016 it ran {fmt((ev.bias_by_year["2016"] - 1) * 100, 1)}% high,
          in 2017 {fmt((1 - ev.bias_by_year["2017"]) * 100, 1)}% low. The pattern is measured here and reported, not hidden.
        </div>
        <div className="table-wrap" style={{ marginTop: 12 }}>
          <table>
            <thead><tr><th>Window</th><th className="r">Forecast ÷ actual</th><th className="r">Year-on-year growth</th></tr></thead>
            <tbody>{growth.map((g) => (
              <tr key={g.name}><td>{g.name}</td><td className="r">{fmt(g.bias, 3)}</td><td className="r">{g.growth == null ? "not available" : `${g.growth > 1 ? "+" : ""}${fmt((g.growth - 1) * 100, 1)}%`}</td></tr>
            ))}</tbody>
          </table>
        </div>
      </section>
    </>
  );
}
