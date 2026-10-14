import PageHead from "@/components/PageHead";
import { BarsH, BarsV, Lines } from "@/components/charts";
import { load } from "@/lib/data";
import { fmt } from "@/lib/format";

export const metadata = { title: "Model | Favorita Demand Forecast" };

const PARAM_INFO = {
  learning_rate: "How big each correction step is",
  max_leaf_nodes: "Maximum size of each tree",
  min_samples_leaf: "Rows needed before a tree can split (guards against noise)",
  l2_regularization: "Penalty on extreme corrections",
  max_features: "Share of features each split may look at",
  n_estimators: "Number of trees (fixed, no early stopping)",
};

export default function ModelPage() {
  const meta = load("meta.json");
  const m = load("models.json");
  const b = load("business.json");
  const ev = load("evaluation.json");

  const cmp = m.phase06_avg.map((x) => ({ name: x.model.replace("28-din", "28-day").replace("(din 1-7 lag>=7)", "(days 1-7, lag 7+)"), value: x.wape, chosen: /HistGB/.test(x.model) }));
  const imp = b.importance.filter((x) => x.kind === "feature").map((x) => ({ name: x.name, value: x.delta_wape, note: x.note }));
  const groups = b.importance.filter((x) => x.kind === "group").map((x) => ({ name: x.name, value: x.delta_wape }));
  const flat = b.importance.filter((x) => x.note);

  const tuneW = meta.windows.filter((w) => w.role === "tuning fold").map((w) => w.window);
  const conf = (m.phase07_confirmation || []).filter((x) => tuneW.includes(x.window));
  const confModels = [...new Set(conf.map((x) => x.model))];
  const confRows = confModels.map((name) => {
    const avg = (cfg) => { const r = conf.filter((x) => x.model === name && x.config === cfg); return r.length ? r.reduce((a, x) => a + x.wape, 0) / r.length : null; };
    return { name, def: avg("default"), tuned: avg("tuned") };
  });

  const trialNames = Object.keys(m.phase07_trials || {});
  const nTrials = Math.max(0, ...trialNames.map((k) => m.phase07_trials[k].length));
  const trialRows = Array.from({ length: nTrials }, (_, i) => {
    const row = { trial: i };
    trialNames.forEach((k) => { row[k] = m.phase07_trials[k][i]?.best ?? null; });
    return row;
  });

  // average the seeds of each variant: "BASE (seed 0)" and "BASE (seed 1)" -> "BASE"
  const ablMap = {};
  (m.phase05_ablation || []).forEach((x) => {
    const k = x.variant.replace(/ \(seed \d+\)$/, "");
    (ablMap[k] = ablMap[k] || []).push(x.wape);
  });
  const abl = Object.entries(ablMap).map(([variant, v]) => ({ variant, wape: v.reduce((a, b) => a + b, 0) / v.length }));
  const base = abl.find((x) => x.variant === "BASE");
  const ablRows = base ? abl.filter((x) => x.variant.startsWith("+") || x.variant.startsWith("-") || x.variant.startsWith("PLACEBO"))
    .map((x) => ({ name: x.variant, value: x.wape - base.wape })).sort((a, b2) => a.value - b2.value) : [];

  const horizon = ev.horizon.map((h) => ({ day: h.day, model: h.wape, naive: h.wape_naive }));
  const sizes = ev.size_groups.map((g) => ({ name: g.group, model: g.wape, naive: g.wape_naive }));

  return (
    <>
      <PageHead
        title="Model"
        intro={`A gradient-boosted tree model (${m.final.model}) reads ${meta.features.length} features per store, family and day and predicts log sales. It was chosen over LightGBM, XGBoost, a linear model and two rules of thumb on the same backtests.`}
        meta={meta}
      />

      <section className="section grid-2">
        <div className="panel">
          <div className="chart-title">Model comparison before tuning</div>
          <div className="chart-note">Average WAPE % over 5 backtest windows. The boosted trees are within noise of each other; all beat the baselines clearly.</div>
          <BarsH data={cmp} xFormat="one" labelFormat="one" highlight="chosen" bars={[{ key: "value", label: "WAPE %", color: "var(--s-forecast)" }]} />
          <p className="note">Highlighted: the model family that won after tuning. The noise threshold measured across seeds is {fmt(meta.noise_wape, 2)} WAPE points.</p>
        </div>
        <div className="panel">
          <div className="chart-title">After tuning, on the 4 tuning folds</div>
          <div className="chart-note">Average WAPE %. Tuning helped every model; HistGB tuned was best and simplest to deploy.</div>
          {confRows.length > 0 && (
            <BarsV data={confRows} yFormat="one" labelFormat="two" yDomain={[0, "auto"]} bars={[
              { key: "def", label: "Default settings", color: "var(--s-naive)" },
              { key: "tuned", label: "Tuned", color: "var(--s-forecast)" },
            ]} />
          )}
          {trialRows.length > 2 && (
            <>
              <div className="chart-title" style={{ marginTop: 20 }}>Best result so far, trial by trial</div>
              <Lines data={trialRows} x="trial" xLabel="Trial" yFormat="two" series={trialNames.map((k, i) => ({
                key: k, label: k, color: ["var(--s-naive)", "var(--s-third)", "var(--s-forecast)"][i], dots: false,
              }))} />
            </>
          )}
        </div>
      </section>

      <h2 className="section-title">What the model relies on</h2>
      <section className="grid-2">
        <div className="panel">
          <div className="chart-title">Feature importance</div>
          <div className="chart-note">How much worse the holdout WAPE gets (points) when one feature is shuffled. Longer bar, more important.</div>
          <BarsH data={imp} xFormat="one" labelFormat="two" bars={[{ key: "value", label: "WAPE increase", color: "var(--s-forecast)" }]} />
          {flat.length > 0 && <p className="note">{flat.map((x) => x.name).join(" and ")}: {flat[0].note.toLowerCase()}.</p>}
        </div>
        <div className="panel">
          <div className="chart-title">By feature group</div>
          <div className="chart-note">Shuffling whole groups at once. Sales history carries most of the signal; the calendar and promotions refine it.</div>
          <BarsH data={groups} xFormat="one" labelFormat="two" bars={[{ key: "value", label: "WAPE increase", color: "var(--s-forecast)" }]} />
          <div className="chart-title" style={{ marginTop: 20 }}>The {meta.features.length} features</div>
          <div className="table-wrap" style={{ marginTop: 8 }}>
            <table>
              <thead><tr><th>Feature</th><th>Group</th><th>Meaning</th></tr></thead>
              <tbody>{meta.features.map((f) => (
                <tr key={f.name}><td><code>{f.name}</code></td><td>{f.group}</td><td className="wrap">{f.description}</td></tr>
              ))}</tbody>
            </table>
          </div>
        </div>
      </section>

      {ablRows.length > 0 && (
        <section className="section panel">
          <div className="chart-title">Feature ideas, tested one at a time</div>
          <div className="chart-note">
            Change in WAPE points against the base feature set when one idea is added (+) or removed (−), averaged over seeds. Below zero helps.
            Gains smaller than the {fmt(meta.noise_wape, 2)}-point noise were treated as no gain. fam_roll28 and the trend features passed alone, but adding them together did not confirm on the follow-up test, so they were left out. Month was kept because it helped clearly on a second window (Aug 2016).
          </div>
          <BarsH data={ablRows} xFormat="two" labelFormat="two" refX={0} bars={[{ key: "value", label: "Change in WAPE", color: "var(--s-forecast)" }]} />
        </section>
      )}

      <h2 className="section-title">Where the error comes from</h2>
      <section className="grid-2">
        <div className="panel">
          <div className="chart-title">Error by forecast day</div>
          <div className="chart-note">WAPE % for day 1 to day 16 after the cutoff, all backtests. Every feature is at least 16 days old, so the model does not degrade with distance.</div>
          <Lines data={horizon} x="day" xLabel="Day ahead" yDomain={[0, "auto"]} series={[
            { key: "model", label: "Model", color: "var(--s-forecast)" },
            { key: "naive", label: "Seasonal naive", color: "var(--s-naive)", dash: true },
          ]} />
        </div>
        <div className="panel">
          <div className="chart-title">Error by series size</div>
          <div className="chart-note">Series grouped by average daily units. Small series have large percentage errors but tiny volume; the largest 20% are {fmt(ev.size_groups[ev.size_groups.length - 1].units_share, 0)}% of units.</div>
          <BarsV data={sizes} yFormat="int" labelFormat="int" bars={[
            { key: "model", label: "Model", color: "var(--s-forecast)" },
            { key: "naive", label: "Seasonal naive", color: "var(--s-naive)" },
          ]} />
        </div>
      </section>

      <section className="section panel">
        <div className="chart-title">Final settings</div>
        <div className="chart-note">Trained once on all data up to {meta.cutoff} ({fmt(meta.training_rows)} rows), seed 0.</div>
        <div className="table-wrap" style={{ marginTop: 10 }}>
          <table>
            <thead><tr><th>Setting</th><th className="r">Value</th><th>What it controls</th></tr></thead>
            <tbody>{Object.entries(m.final.params).map(([k, v]) => (
              <tr key={k}><td><code>{k}</code></td><td className="r">{typeof v === "number" && !Number.isInteger(v) ? fmt(v, 4) : fmt(v)}</td><td>{PARAM_INFO[k] || ""}</td></tr>
            ))}</tbody>
          </table>
        </div>
      </section>
    </>
  );
}
