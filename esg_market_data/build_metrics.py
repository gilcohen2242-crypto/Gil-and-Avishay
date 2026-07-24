from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

START = "2021-01-01"
END = "2026-07-16"  # yfinance end is exclusive; includes 2026-07-15
TRADING_DAYS = 252
RF_ANNUAL = 0.0
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)


def normalize_ticker(t: str) -> str:
    return t.replace(".", "-")


def metrics_for(series: pd.Series, spy: pd.Series) -> dict:
    s = series.dropna().astype(float)
    if len(s) < 2:
        return {"status": "insufficient data", "observations": len(s)}
    r = s.pct_change().dropna()
    years = (s.index[-1] - s.index[0]).days / 365.25
    total_return = s.iloc[-1] / s.iloc[0] - 1
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if years > 0 and s.iloc[0] > 0 else np.nan
    vol = r.std(ddof=1) * math.sqrt(TRADING_DAYS)
    downside = r[r < 0]
    downside_dev = downside.std(ddof=1) * math.sqrt(TRADING_DAYS) if len(downside) > 1 else np.nan
    ann_return = r.mean() * TRADING_DAYS
    sharpe = (ann_return - RF_ANNUAL) / vol if vol and not np.isnan(vol) else np.nan
    sortino = (ann_return - RF_ANNUAL) / downside_dev if downside_dev and not np.isnan(downside_dev) else np.nan
    wealth = (1 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    max_dd = drawdown.min()
    var95 = r.quantile(0.05)
    cvar95 = r[r <= var95].mean()
    common = pd.concat([r.rename("stock"), spy.pct_change().rename("spy")], axis=1).dropna()
    beta = alpha = corr = np.nan
    if len(common) > 20 and common["spy"].var(ddof=1) != 0:
        beta = common["stock"].cov(common["spy"]) / common["spy"].var(ddof=1)
        alpha = (common["stock"].mean() - beta * common["spy"].mean()) * TRADING_DAYS
        corr = common["stock"].corr(common["spy"])
    annual = {}
    for year in range(2021, 2027):
        y = s[s.index.year == year]
        annual[str(year)] = y.iloc[-1] / y.iloc[0] - 1 if len(y) >= 2 else np.nan
    return {
        "status": "ok",
        "first_date": s.index[0].date().isoformat(),
        "last_date": s.index[-1].date().isoformat(),
        "observations": len(s),
        "start_price": s.iloc[0],
        "end_price": s.iloc[-1],
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": vol,
        "downside_deviation": downside_dev,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "var95_daily": var95,
        "cvar95_daily": cvar95,
        "beta_spy": beta,
        "alpha_annual_spy": alpha,
        "correlation_spy": corr,
        "best_day": r.max(),
        "worst_day": r.min(),
        "positive_days_pct": (r > 0).mean(),
        **{f"return_{y}": annual[str(y)] for y in range(2021, 2027)},
    }


def main() -> None:
    original = [x.strip() for x in (ROOT / "tickers_exact.txt").read_text().splitlines() if x.strip()]
    mapped = {t: normalize_ticker(t) for t in original}
    download = sorted(set(mapped.values()) | {"SPY"})
    chunks = [download[i:i+100] for i in range(0, len(download), 100)]
    frames = []
    failures = []
    for i, chunk in enumerate(chunks, start=1):
        try:
            d = yf.download(chunk, start=START, end=END, auto_adjust=True, progress=False, group_by="column", threads=True)
            close = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d[["Close"]].rename(columns={"Close": chunk[0]})
            frames.append(close)
        except Exception as exc:
            failures.append({"chunk": i, "error": repr(exc), "tickers": chunk})
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    spy = prices["SPY"].dropna() if "SPY" in prices else pd.Series(dtype=float)
    rows = []
    daily_rows = []
    for t in original:
        yt = mapped[t]
        if yt in prices:
            s = prices[yt].dropna()
            m = metrics_for(s, spy)
            for dt, price in s.items():
                daily_rows.append({"ticker": t, "date": dt.date().isoformat(), "adjusted_close": float(price)})
        else:
            m = {"status": "not downloaded", "observations": 0}
        rows.append({"ticker": t, "yahoo_ticker": yt, **m})
    pd.DataFrame(rows).to_csv(OUT / "risk_return_metrics.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(OUT / "daily_adjusted_prices.csv", index=False)
    (OUT / "run_metadata.json").write_text(json.dumps({
        "requested_start": START,
        "requested_last_date": "2026-07-15",
        "benchmark": "SPY",
        "risk_free_rate_annual": RF_ANNUAL,
        "ticker_count": len(original),
        "download_symbol_count": len(download),
        "failures": failures,
    }, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
