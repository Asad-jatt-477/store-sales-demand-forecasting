import "@fontsource/schibsted-grotesk/400.css";
import "@fontsource/schibsted-grotesk/600.css";
import "@fontsource/schibsted-grotesk/700.css";
import "./globals.css";
import Nav from "@/components/Nav";
import { load } from "@/lib/data";

export const metadata = {
  title: "Favorita Demand Forecast",
  description: "16-day demand forecasting for 1,782 store-family series of Corporación Favorita (Ecuador): live what-if forecasts, backtests, model and business evaluation.",
};

export default function RootLayout({ children }) {
  const meta = load("meta.json");
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <Nav meta={meta} />
          <main className="main">{children}</main>
        </div>
      </body>
    </html>
  );
}
