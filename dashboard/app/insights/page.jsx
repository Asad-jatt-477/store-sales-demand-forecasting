import PageHead from "@/components/PageHead";
import { TimeChart, BarsH, BarsV, Lines } from "@/components/charts";
import { load } from "@/lib/data";
import { fmt, compact } from "@/lib/format";

export const metadata = { title: "Data insights | Favorita Demand Forecast" };

// weekly average of daily totals: the raw daily line is too spiky to read across 4.5 years
function weekly(daily) {
  const out = [];
  for (let i = 0; i < daily.length; i += 7) {
    const chunk = daily.slice(i, i + 7);
    if (chunk.length < 7) break;
    out.push({ date: chunk[0].date, sales: chunk.reduce((a, d) => a + d.sales, 0) / chunk.length });
  }
  return out;
}

export default function Insights() {
  const meta = load("meta.json");
  const ins = load("insights.json");
  const wk = weekly(ins.daily_total);
  const cleanWeek = wk.find((d) => d.date >= ins.clean_start)?.date;
  const years = ins.yearly.map((y) => ({ name: y.year === 2017 ? "2017 (to Aug)" : String(y.year), value: y.per_day }));
  const growth = ((ins.yearly[ins.yearly.length - 1].per_day / ins.yearly[ins.yearly.length - 2].per_day) - 1) * 100;

  const quake = ins.earthquake;
  const qOn = quake.filter((d) => d.flag === 1);
  const school = Object.keys(ins.school_season);
  const schoolRows = ins.school_season[school[0]].map((_, i) => {
    const row = { week: i + 1 };
    school.forEach((y) => { row[y] = ins.school_season[y][i]?.sales ?? null; });
    return row;
  });
  const schoolPeak = school.map((y) => ins.school_season[y].reduce((a, b) => (b.sales > a.sales ? b : a)));

  const famShare = ins.family_share.slice(0, 15).map((f) => ({ name: f.family, value: f.share }));
  const top5 = ins.family_share.slice(0, 5).reduce((a, f) => a + f.share, 0);
  const sparse = [...ins.family_share].sort((a, b) => b.zero_days - a.zero_days).slice(0, 8);
  const lift = ins.promo_lift.filter((p) => p.family !== "SCHOOL AND OFFICE SUPPLIES").slice(0, 16).map((p) => ({ name: p.family, value: p.lift }));
  const schoolLift = ins.promo_lift.find((p) => p.family === "SCHOOL AND OFFICE SUPPLIES");
  const stores = ins.store_totals.slice(0, 12).map((s) => ({ name: `Store ${s.store}, ${s.city}`, value: s.sales }));
  const o = ins.outliers;
  const causes = Object.entries(o.causes || {}).map(([k, v]) => ({ name: k, value: v })).sort((a, b) => b.value - a.value);
  const causesTotal = causes.reduce((a, c) => a + c.value, 0);
  const tests = Object.entries(o.tests || {});
  const methods = tests.length ? Object.keys(tests[0][1]) : [];
  const methodName = { "Raw (seed 1, noise check)": "Raw, seed 1", "Cap (H5)": "Spikes capped", "Replace (L4 + L0)": "Drops replaced" };

  return (
    <>
      <PageHead
        title="Data insights"
        intro={`What 4.5 years of sales say about how Favorita customers buy: ${meta.n_stores} stores, ${meta.n_families} product families, ${meta.history_start} to ${meta.cutoff}. Every pattern here was checked before it was allowed into the model.`}
        meta={meta}
      />

      <section className="section panel">
        <div className="chart-title">Average units sold per day, week by week</div>
        <div className="chart-note">
          Sales grew steadily. Before {ins.clean_start} several product families were not recorded at all, so the model trains only on the period after the dashed line.
        </div>
        <TimeChart data={wk} height={300} xFormat="month" series={[{ key: "sales", label: "Units per day", color: "var(--s-actual)", width: 1.6 }]}
          marker={cleanWeek} markerLabel="Training data starts" />
      </section>

      <section className="section grid-2">
        <div className="panel">
          <div className="chart-title">Units per day by year</div>
          <div className="chart-note">2017 so far is {fmt(growth, 1)}% above the 2016 daily rate. Growth is why the model sometimes under-forecasts.</div>
          <BarsV data={years} yFormat="compact" labelFormat="compact" bars={[{ key: "value", label: "Units per day", color: "var(--s-forecast)" }]} />
        </div>
        <div className="panel">
          <div className="chart-title">Share of units by store type</div>
          <div className="chart-note">Types A and D carry two thirds of the volume. Store type itself is not a model input: removing it changed the error by less than the noise.</div>
          <BarsV data={ins.store_type_share.map((t) => ({ name: `Type ${t.type}`, value: t.share }))} yFormat="pct" labelFormat="pct"
            bars={[{ key: "value", label: "Share of units", color: "var(--s-forecast)" }]} />
        </div>
      </section>

      <h2 className="section-title">Rhythms the model learns</h2>
      <section className="grid-2">
        <div className="panel">
          <div className="chart-title">Day of week</div>
          <div className="chart-note">Index: 100 is an average day. Saturday and Sunday sell 20–28% above average; Thursday is the quietest day.</div>
          <BarsV data={ins.dow_index.map((d) => ({ name: d.day, value: d.index }))} yFormat="idx" labelFormat="idx" refY={100} yDomain={[0, "auto"]}
            bars={[{ key: "value", label: "Index", color: "var(--s-forecast)" }]} />
        </div>
        <div className="panel">
          <div className="chart-title">Month of year</div>
          <div className="chart-note">December is the clear peak; the other months stay within about 7% of average.</div>
          <BarsV data={ins.month_index.map((d) => ({ name: d.month, value: d.index }))} yFormat="idx" refY={100}
            bars={[{ key: "value", label: "Index", color: "var(--s-forecast)" }]} />
        </div>
      </section>
      <section className="section panel">
        <div className="chart-title">Day of month</div>
        <div className="chart-note">Sales are highest from the 30th to the 4th and dip mid-month and in the last week before month end: a payday rhythm the model reads from the day of the month.</div>
        <BarsV data={ins.dom_index.map((d) => ({ name: d.day, value: d.index }))} yFormat="idx" refY={100} height={220}
          bars={[{ key: "value", label: "Index", color: "var(--s-forecast)" }]} />
      </section>

      <h2 className="section-title">Products and stores</h2>
      <section className="grid-2">
        <div className="panel">
          <div className="chart-title">Share of units, top 15 families</div>
          <div className="chart-note">The top 5 families are {fmt(top5, 0)}% of all units, so they decide most of the total error.</div>
          <BarsH data={famShare} xFormat="pct" labelFormat="pct" bars={[{ key: "value", label: "Share of units", color: "var(--s-forecast)" }]} />
        </div>
        <div className="panel">
          <div className="chart-title">Biggest stores by units sold</div>
          <div className="chart-note">Since training data starts. Quito stores dominate.</div>
          <BarsH data={stores} xFormat="compact" labelFormat="compact" bars={[{ key: "value", label: "Units", color: "var(--s-forecast)" }]} />
          <div className="chart-title" style={{ marginTop: 22 }}>Families that often sell nothing</div>
          <div className="chart-note">Share of store-days with zero sales. Sparse series are hard to forecast in percentage terms.</div>
          <div className="table-wrap" style={{ marginTop: 8 }}>
            <table>
              <thead><tr><th>Family</th><th className="r">Zero-sale days</th><th className="r">Share of units</th></tr></thead>
              <tbody>{sparse.map((f) => (
                <tr key={f.family}><td>{f.family}</td><td className="r">{fmt(f.zero_days, 1)}%</td><td className="r">{fmt(f.share, 2)}%</td></tr>
              ))}</tbody>
            </table>
          </div>
        </div>
      </section>

      <h2 className="section-title">Promotions, holidays and shocks</h2>
      <section className="grid-2">
        <div className="panel">
          <div className="chart-title">Sales on promotion days vs other days</div>
          <div className="chart-note">
            Average sales when items were on promotion, % above days without. This is an association: promotions are planned for busy seasons.
          </div>
          <BarsH data={lift} xFormat="int" labelFormat="pp" bars={[{ key: "value", label: "% higher on promotion", color: "var(--s-forecast)" }]} />
          {schoolLift && (
            <p className="note" style={{ marginTop: 8 }}>
              School and office supplies is left out: +{fmt(schoolLift.lift, 0)}%, because its promotions fall in the back-to-school season, not because promotions alone cause that jump.
            </p>
          )}
        </div>
        <div className="panel">
          <div className="chart-title">School and office supplies, July to September</div>
          <div className="chart-note">Weekly units. The back-to-school peak repeats every year, which is why lag_364 and the season matter.</div>
          <Lines data={schoolRows} x="week" xLabel="Week from 1 July" yFormat="compact" series={school.map((y, i) => ({
            key: y, label: y, color: ["var(--s-naive)", "var(--s-third)", "var(--s-forecast)"][i] || "var(--s-actual)",
          }))} />
          <p className="note" style={{ marginTop: 8 }}>
            Peak week: {schoolPeak.map((p, i) => `${school[i]} week ${p.week} (${compact(p.sales)})`).join(", ")}.
          </p>
          <div className="figures" style={{ marginTop: 18 }}>
            <div className="figure">
              <div className="figure-value">{ins.holiday_effect_pct > 0 ? "+" : ""}{fmt(ins.holiday_effect_pct, 1)}%</div>
              <div className="figure-label">Sales on national holidays vs the same weekday without one</div>
            </div>
          </div>
        </div>
      </section>

      <section className="section panel">
        <div className="chart-title">The April 2016 earthquake</div>
        <div className="chart-note">
          After the earthquake on 16 April 2016, people bought water and supplies in bulk. Those weeks are flagged so the model does not learn them as normal.
        </div>
        <TimeChart data={quake} height={260} series={[{ key: "sales", label: "Units per day", color: "var(--s-actual)", width: 1.6 }]}
          shadeFrom={qOn[0]?.date} shadeTo={qOn[qOn.length - 1]?.date} marker={qOn[0]?.date} markerLabel="Earthquake" />
      </section>

      <h2 className="section-title">Unusual days</h2>
      <section className="grid-outlier">
        <div className="panel">
          <div className="figures">
            <div className="figure"><div className="figure-value">{fmt(o.robust_z)}</div><div className="figure-label">Extreme days found</div>
              <div className="figure-sub">{o.rows ? `${fmt((o.robust_z / o.rows) * 100, 2)}% of ${compact(o.rows)} rows` : ""}</div></div>
            <div className="figure"><div className="figure-value">{fmt(o.rows_changed ?? 0)}</div><div className="figure-label">Rows changed</div>
              <div className="figure-sub">Kept as real demand</div></div>
          </div>
          <div className="chart-title" style={{ marginTop: 20 }}>What caused them</div>
          <div className="chart-note">Each extreme day was traced to a cause before deciding what to do with it.</div>
          <div className="table-wrap" style={{ marginTop: 8 }}>
            <table>
              <thead><tr><th>Cause</th><th className="r">Days</th><th className="r">Share</th></tr></thead>
              <tbody>{causes.map((c) => (
                <tr key={c.name}><td>{c.name}</td><td className="r">{fmt(c.value)}</td><td className="r">{fmt((c.value / causesTotal) * 100, 1)}%</td></tr>
              ))}</tbody>
            </table>
          </div>
        </div>
        <div className="panel">
          <div className="chart-title">Does fixing them help? Tested, not assumed</div>
          <div className="chart-note">Model error (WAPE %) with each treatment. Raw, seed 1 is the same raw data with another random seed: it shows how much results move by chance. No treatment lowered the error by more than that, and capping spikes made the post-earthquake window worse, so the raw data was kept.</div>
          <div className="table-wrap" style={{ marginTop: 12 }}>
            <table>
              <thead><tr><th>Window</th>{methods.map((m) => <th key={m} className="r">{methodName[m] || m}</th>)}</tr></thead>
              <tbody>{tests.map(([w, v]) => (
                <tr key={w}><td>{w}</td>{methods.map((m) => <td key={m} className="r">{fmt(v[m], 2)}</td>)}</tr>
              ))}</tbody>
            </table>
          </div>
        </div>
      </section>
    </>
  );
}
