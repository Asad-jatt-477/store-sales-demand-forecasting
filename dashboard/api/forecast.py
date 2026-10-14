"""
Live what-if forecast API (Vercel Python function)  ->  POST /api/forecast

Body (JSON):
  {"store": 44, "family": "BEVERAGES",
   "promo":   [16 numbers] or null   # items on promotion per day (null = the planned values from test.csv)
   "holiday": [16 0/1]     or null}  # national holiday flag per day (null = the real calendar)

Returns the model's 16-day forecast for 16-31 Aug 2017 twice: with the real inputs ("baseline") and with the
user's inputs ("scenario"), plus safety-buffer bands learned in Phase 09.
Only promotion and holiday can be changed: the other 13 features are sales history (lags) and the calendar,
which are fixed facts on the forecast date.
"""
import gzip
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ASSETS = Path(__file__).resolve().parent / "assets"
_STATE = {}


def _load():
    if _STATE:
        return _STATE
    meta = json.loads((ASSETS / "meta.json").read_text())
    with gzip.open(ASSETS / "test_features.csv.gz", "rt", encoding="utf-8") as fh:
        tf = pd.read_csv(fh)
    tf["family"] = tf["family"].astype(str)
    _STATE.update(meta=meta, tf=tf, model=joblib.load(ASSETS / "model.joblib"))
    return _STATE


def _frame(rows, meta):
    X = rows[meta["features"]].copy()
    for c in meta["categorical"]:
        cats = pd.Index(meta["categories"][c])
        if meta["category_dtype"][c].startswith("int"):
            cats = cats.astype(meta["category_dtype"][c])
            vals = X[c].astype(meta["category_dtype"][c])
        else:
            vals = X[c].astype(str)
        X[c] = pd.Categorical(vals, categories=cats)
    return X


def forecast(store, family, promo=None, holiday=None):
    s = _load()
    meta, tf, model = s["meta"], s["tf"], s["model"]
    if family not in meta["categories"]["family_cat"]:
        raise ValueError(f"unknown family: {family}")
    rows = tf[(tf["store_nbr"] == int(store)) & (tf["family"] == family)].sort_values("date")
    if len(rows) != 16:
        raise ValueError(f"unknown store: {store}")
    zero = rows["zero"].values == 1
    base_X = _frame(rows, meta)
    scen_X = base_X.copy()
    if promo is not None:
        if len(promo) != 16 or any((not isinstance(v, (int, float))) or v < 0 or v > 10000 for v in promo):
            raise ValueError("promo must be 16 numbers between 0 and 10000")
        scen_X["onpromotion"] = np.array(promo, dtype=float)
    if holiday is not None:
        if len(holiday) != 16:
            raise ValueError("holiday must be 16 values (0/1)")
        scen_X["is_national_holiday"] = np.array([1 if v else 0 for v in holiday], dtype=scen_X["is_national_holiday"].dtype)
    base = np.where(zero, 0, np.expm1(model.predict(base_X)).clip(0))
    scen = np.where(zero, 0, np.expm1(model.predict(scen_X)).clip(0))
    b = meta["buffers"].get(family, {"p80": 0, "p90": 0})
    rnd = lambda a: [round(float(v), 2) for v in a]
    return {"store": int(store), "family": family, "dates": rows["date"].tolist(),
            "baseline": rnd(base), "scenario": rnd(scen),
            "scenario_p80": rnd(scen * (1 + b["p80"] / 100)), "scenario_p90": rnd(scen * (1 + b["p90"] / 100)),
            "buffer_pct": b, "never_sold": bool(zero.all()),
            "inputs": {"promo_planned": rnd(rows["onpromotion"].values), "holiday_planned": [int(v) for v in rows["is_national_holiday"].values],
                       "promo_used": rnd(scen_X["onpromotion"].values), "holiday_used": [int(v) for v in scen_X["is_national_holiday"].values]},
            "totals": {"baseline": round(float(base.sum()), 1), "scenario": round(float(scen.sum()), 1),
                       "change_pct": round(float((scen.sum() / base.sum() - 1) * 100), 2) if base.sum() > 0 else 0.0}}


class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            s = _load()
            self._send(200, {"status": "ok", "series": int(len(s["tf"]) // 16), "window": s["meta"]["forecast_window"]})
        except Exception as e:  # pragma: no cover
            self._send(500, {"error": str(e)})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, forecast(req.get("store"), req.get("family"), req.get("promo"), req.get("holiday")))
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # pragma: no cover
            self._send(500, {"error": f"server error: {e}"})
