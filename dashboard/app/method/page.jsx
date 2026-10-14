import PageHead from "@/components/PageHead";
import { load } from "@/lib/data";
import { fmt } from "@/lib/format";

export const metadata = { title: "Method | Favorita Demand Forecast" };

export default function Method() {
  const meta = load("meta.json");
  const w = meta.windows;
  return (
    <>
      <PageHead
        title="Method"
        intro="How the forecast was built and tested, the decisions that shaped it, and what it cannot do. Every choice below was decided by a measured result on held-out data, not by default."
        meta={meta}
      />

      <section className="section grid-2">
        <div className="panel">
          <div className="chart-title">The forecasting task</div>
          <ul className="list" style={{ marginTop: 10 }}>
            <li>Predict daily unit sales for {fmt(meta.n_series)} store and family series, {meta.horizon} days ahead ({meta.forecast_window[0]} to {meta.forecast_window[1]}).</li>
            <li>On the forecast date, only sales up to {meta.cutoff} are known. Promotions are planned in advance, so the test file includes them.</li>
            <li>Target: log(1 + units). Errors are reported as WAPE (absolute error ÷ units sold) and RMSLE (the Kaggle metric).</li>
            <li>Every result is compared with a seasonal-naive rule: the average of the same weekday in recent weeks.</li>
          </ul>
        </div>
        <div className="panel">
          <div className="chart-title">No peeking into the future</div>
          <ul className="list" style={{ marginTop: 10 }}>
            <li>Every sales feature is at least 16 days old, so day 16 of the forecast never uses a sale the model could not know.</li>
            <li>Each backtest retrains the model only on data before its cutoff and rebuilds the features from that point.</li>
            <li>The holdout window was not used for any choice: not features, not model, not settings, not buffers.</li>
            <li>Safety buffers were learned on earlier windows and only checked on the holdout.</li>
          </ul>
        </div>
      </section>

      <section className="section panel">
        <div className="chart-title">Backtest design</div>
        <div className="chart-note">Six 16-day windows, chosen to cover normal weeks, the post-earthquake period and the pre-Christmas rush.</div>
        <div className="table-wrap" style={{ marginTop: 12 }}>
          <table>
            <thead><tr><th>Window</th><th>Training data until</th><th>Forecast from</th><th>Used for</th><th className="r">Model WAPE</th><th className="r">Naive WAPE</th></tr></thead>
            <tbody>{w.map((x) => (
              <tr key={x.window}><td>{x.window}</td><td>{x.cutoff}</td><td>{x.start}</td><td>{x.role}</td><td className="r">{fmt(x.wape, 2)}%</td><td className="r">{fmt(x.wape_naive, 2)}%</td></tr>
            ))}</tbody>
          </table>
        </div>
        <p className="note" style={{ marginTop: 12 }}>
          Noise check: the same model with a different random seed moves WAPE by up to {fmt(meta.noise_wape, 2)} points. Differences smaller than that were treated as ties, and the simpler option won.
        </p>
      </section>

      <section className="section grid-2">
        <div className="panel">
          <div className="chart-title">Decisions, and the evidence behind them</div>
          <ul className="list" style={{ marginTop: 10 }}>
            <li><b>Training starts {meta.clean_start}.</b> Before that, some families were not recorded at all; zeros there mean missing data, not zero demand.</li>
            <li><b>Outliers kept.</b> Capping spikes or replacing drops did not reduce error beyond noise, and most extreme days traced back to real events such as company-wide special days and promotions.</li>
            <li><b>{meta.features.length} features.</b> Each candidate feature was added or removed one at a time; ideas that did not beat the noise were dropped.</li>
            <li><b>HistGradientBoosting.</b> Tied with LightGBM and XGBoost before tuning, best after tuning, and needs only scikit-learn to run.</li>
            <li><b>Fixed {meta.params.n_estimators} trees, no early stopping.</b> Early stopping would have used validation data inside training.</li>
            <li><b>Zero rule.</b> Series that had never sold, or stores not yet open, are forecast as 0.</li>
          </ul>
        </div>
        <div className="panel">
          <div className="chart-title">Limits to keep in mind</div>
          <ul className="list" style={{ marginTop: 10 }}>
            <li>Local and regional holidays are not model inputs; error is higher on those days.</li>
            <li>When the business grows fast the forecast runs a few percent low (2017), and high when growth slows (2016).</li>
            <li>Very small series (under 1 unit a day) have large percentage errors, though tiny volume.</li>
            <li>Promotion effects are associations learned from history. Promotions fall in busy seasons, so the what-if studio shows what usually happened, not a guaranteed cause and effect.</li>
            <li>Oil price showed almost no link to sales once the trend was removed, and transactions are not in the test file, so neither is used. Store type, cluster and city were removed after testing: taking them out changed the error by less than the noise.</li>
          </ul>
        </div>
      </section>

      <section className="section panel">
        <div className="chart-title">Reproduce it</div>
        <ul className="list" style={{ marginTop: 10 }}>
          <li>Data: Kaggle, Store Sales: Time Series Forecasting (Corporación Favorita).</li>
          <li>Pipeline: ten Python scripts in <code>src/</code>, run in order from the project root. Script 10 exports everything this site shows.</li>
          <li>Model: scikit-learn {meta.versions["scikit-learn"]}, pandas {meta.versions.pandas}, numpy {meta.versions.numpy}. The live forecast runs the same saved model as a Vercel Python function.</li>
          <li>Site data generated {meta.generated}.</li>
        </ul>
      </section>
    </>
  );
}
