import PageHead from "@/components/PageHead";
import ForecastStudio from "@/components/ForecastStudio";
import { load } from "@/lib/data";

export const metadata = { title: "Forecast studio | Favorita Demand Forecast" };

export default function ForecastPage() {
  const meta = load("meta.json");
  return (
    <>
      <PageHead
        title="Forecast studio"
        intro="Pick a store and a product family, change the promotion plan or mark holidays, and the trained model recalculates the 16-day forecast live. Sales history and the calendar stay fixed: on the forecast date they are facts, not choices."
        meta={meta}
      />
      <ForecastStudio stores={meta.stores} families={meta.families} window={meta.forecast_window} cutoff={meta.cutoff} />
      <section className="section">
        <div className="panel">
          <div className="chart-title">How to read this</div>
          <ul className="list" style={{ marginTop: 10 }}>
            <li><b>Forecast with planned inputs</b> uses the promotion plan that came with the data and the real holiday calendar.</li>
            <li><b>Forecast with your inputs</b> appears when you change something and press Run forecast.</li>
            <li><b>Order for 80% / 90%</b> adds the family&apos;s safety buffer, learned from past backtests, so stock covers that share of days.</li>
            <li>The model learned how promotions and holidays moved sales in the past. It is an association, not a guarantee of cause and effect.</li>
          </ul>
        </div>
      </section>
    </>
  );
}
