/** @type {import('next').NextConfig} */
const nextConfig = {
  // Local development: the Python forecast function runs separately (python api/dev_server.py).
  // On Vercel, /api/forecast is served by api/forecast.py automatically.
  async rewrites() {
    if (process.env.NODE_ENV !== "development") return [];
    return [{ source: "/api/:path*", destination: "http://127.0.0.1:8000/api/:path*" }];
  },
};
export default nextConfig;
