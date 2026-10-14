"""Local development only: serves the Vercel function at http://127.0.0.1:8000/api/forecast
Run from the dashboard folder:  python api/dev_server.py   (then `npm run dev` in another terminal)."""
import sys
from http.server import HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast import handler  # noqa: E402

if __name__ == "__main__":
    print("Forecast API: http://127.0.0.1:8000/api/forecast")
    HTTPServer(("127.0.0.1", 8000), handler).serve_forever()
