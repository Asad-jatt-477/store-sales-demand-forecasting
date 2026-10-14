"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const I = {
  overview: <path d="M3 13h4v6H3zM10 8h4v11h-4zM17 4h4v15h-4z" />,
  forecast: <path d="M3 17l5-5 4 3 8-9M15 6h5v5" />,
  backtests: <path d="M4 4v16h16M8 14l3-3 3 2 5-6" />,
  insights: <path d="M4 19V9M10 19V5M16 19v-7M22 19H2" />,
  model: <path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M6 18l2.5-2.5M15.5 8.5L18 6M12 9a3 3 0 100 6 3 3 0 000-6z" />,
  business: <path d="M3 7h18v12H3zM8 7V5h8v2M3 12h18" />,
  method: <path d="M5 4h11l3 3v13H5zM9 10h7M9 14h7M9 18h4" />,
};
const items = [
  ["/", "Overview", "overview"],
  ["/forecast", "Forecast studio", "forecast"],
  ["/backtests", "Backtests", "backtests"],
  ["/insights", "Data insights", "insights"],
  ["/model", "How the model works", "model"],
  ["/business", "Business impact", "business"],
  ["/method", "Method & limits", "method"],
];

export default function Nav({ meta }) {
  const path = usePathname();
  return (
    <aside className="side">
      <div className="brand">
        <div className="brand-mark">
          <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">
            <rect width="26" height="26" rx="6" fill="var(--brand)" />
            <path d="M6 17l4-4 3 2 7-7" stroke="var(--brand-ink)" strokeWidth="2.2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Favorita Demand
        </div>
        <div className="brand-sub">16-day store demand forecasting for a grocery chain in Ecuador</div>
      </div>
      <nav className="nav" aria-label="Main">
        {items.map(([href, label, icon]) => (
          <Link key={href} href={href} aria-current={path === href ? "page" : undefined}>
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{I[icon]}</svg>
            {label}
          </Link>
        ))}
      </nav>
      <div className="side-foot">
        <div><b>{meta.model}</b> ({meta.model_config})</div>
        <div>Data to {meta.cutoff}</div>
        <div>Forecast {meta.forecast_window[0]} – {meta.forecast_window[1]}</div>
        <div style={{ marginTop: 8 }}>Built by </div>
      </div>
    </aside>
  );
}
