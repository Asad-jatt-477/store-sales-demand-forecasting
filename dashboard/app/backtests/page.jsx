import PageHead from "@/components/PageHead";
import BacktestExplorer from "@/components/BacktestExplorer";
import { BarsV } from "@/components/charts";
import { load } from "@/lib/data";
import { fmt } from "@/lib/format";

export const metadata = { title: "Backtests | Favorita Demand Forecast" };

export default function Backtests() {
  const meta = load("meta.json");
  const ev = load("evaluation.json");
  const w = ev.windows;
  const bars = w.map((x) => ({ name: x.window.replace("Validation (Aug 2017)", "Validation").replace(/^Holdout.*/, "Holdout"), model: x.wape, naive: x.wape_naive }));
  return (
    <>
      <PageHead
        title="Backtests"
        intro="The honest way to test a forecaster is to go back in time: train only on what was known on a past date and compare the next 16 days with what really sold. Six such windows were run; the holdout window was never used for any decision."
        meta={meta}
      />
      <BacktestExplorer stores={meta.stores} families={meta.families} windows={w} />

      <section className="section panel">
        <div>
          <div className="chart-title">Error per window, all stores and families</div>
          <div className="chart-note">WAPE: total absolute error as a share of units sold. Lower is better.</div>
          <BarsV data={bars} height={260} yFormat="one" labelFormat="one" bars={[
            { key: "model", label: "Model", color: "var(--s-forecast)" },
            { key: "naive", label: "Seasonal naive", color: "var(--s-naive)" },
          ]} />
        </div>
        <div style={{ marginTop: 26 }}>
          <div className="chart-title">Window details</div>
          <div className="chart-note">Forecast / actual above 1 means the model over-forecast that window</div>
          <div className="table-wrap" style={{ marginTop: 12 }}>
            <table>
              <thead><tr><th>Window</th><th>Used for</th><th className="r">WAPE</th><th className="r">Naive</th><th className="r">RMSLE</th><th className="r">Fcst / actual</th></tr></thead>
              <tbody>
                {w.map((x) => (
                  <tr key={x.window}>
                    <td>{x.window}</td><td>{x.role}</td><td className="r">{fmt(x.wape, 2)}%</td><td className="r">{fmt(x.wape_naive, 2)}%</td>
                    <td className="r">{fmt(x.rmsle, 3)}</td><td className="r">{fmt(x.bias, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note" style={{ marginTop: 12 }}>
            Tuning folds chose the settings. The validation window decided between models. The holdout ({w[w.length - 1].start}) was only reported, so it is the most honest estimate.
          </p>
        </div>
      </section>
    </>
  );
}
