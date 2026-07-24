from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def clean_ticker(t: str) -> str:
    t = t.strip().upper()
    return t.replace("BRK.B", "BRK-B").replace("CWEN.A", "CWEN-A")


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def get_row(stmt: pd.DataFrame, names: list[str]):
    if stmt is None or stmt.empty:
        return None
    for name in names:
        if name in stmt.index:
            return pd.to_numeric(stmt.loc[name], errors="coerce")
    return None


def cagr(values: pd.Series, years: int = 3):
    s = values.dropna().sort_index()
    if len(s) < years + 1:
        return np.nan
    end = float(s.iloc[-1])
    start = float(s.iloc[-(years + 1)])
    if start <= 0 or end <= 0:
        return np.nan
    return (end / start) ** (1 / years) - 1


def fetch_one(raw_ticker: str):
    ticker = clean_ticker(raw_ticker)
    summary = {"Ticker": raw_ticker, "Yahoo_Ticker": ticker, "Status": "No data"}
    annual_rows = []
    try:
        tk = yf.Ticker(ticker)
        info = tk.get_info() or {}
        summary.update({
            "Company_Name_Yahoo": info.get("longName") or info.get("shortName"),
            "Currency": info.get("currency"),
            "Market_Cap": num(info.get("marketCap")),
            "Trailing_PE": num(info.get("trailingPE")),
            "Forward_PE": num(info.get("forwardPE")),
            "PEG_Ratio": num(info.get("pegRatio") or info.get("trailingPegRatio")),
            "Price_to_Sales": num(info.get("priceToSalesTrailing12Months")),
            "Revenue_Growth_TTM_YoY": num(info.get("revenueGrowth")),
            "Earnings_Growth_TTM_YoY": num(info.get("earningsGrowth")),
            "EPS_Growth_Quarterly_YoY": num(info.get("earningsQuarterlyGrowth")),
            "Profit_Margin": num(info.get("profitMargins")),
            "Data_Source": "Yahoo Finance via yfinance",
        })

        stmt = tk.get_income_stmt(freq="yearly")
        rev = get_row(stmt, ["Total Revenue", "Operating Revenue"])
        ni = get_row(stmt, ["Net Income", "Net Income Common Stockholders"])
        if rev is not None or ni is not None:
            dates = sorted(set((rev.index if rev is not None else [])).union(ni.index if ni is not None else []))
            rev_series = pd.Series({d: num(rev.get(d)) if rev is not None else np.nan for d in dates})
            ni_series = pd.Series({d: num(ni.get(d)) if ni is not None else np.nan for d in dates})
            for d in dates:
                annual_rows.append({
                    "Ticker": raw_ticker,
                    "Yahoo_Ticker": ticker,
                    "Fiscal_Year_End": pd.Timestamp(d).date().isoformat(),
                    "Revenue": rev_series.get(d, np.nan),
                    "Net_Income": ni_series.get(d, np.nan),
                })
            rev_sorted = rev_series.dropna().sort_index()
            ni_sorted = ni_series.dropna().sort_index()
            summary["Revenue_Latest_FY"] = rev_sorted.iloc[-1] if len(rev_sorted) else np.nan
            summary["Net_Income_Latest_FY"] = ni_sorted.iloc[-1] if len(ni_sorted) else np.nan
            summary["Revenue_Growth_Latest_FY"] = rev_sorted.pct_change().iloc[-1] if len(rev_sorted) >= 2 else np.nan
            summary["Net_Income_Growth_Latest_FY"] = ni_sorted.pct_change().iloc[-1] if len(ni_sorted) >= 2 and ni_sorted.iloc[-2] != 0 else np.nan
            summary["Revenue_CAGR_3Y"] = cagr(rev_sorted, 3)
            summary["Net_Income_CAGR_3Y"] = cagr(ni_sorted, 3)
            summary["Latest_Fiscal_Year"] = max(dates).date().isoformat() if dates else None

        useful = [summary.get("Trailing_PE"), summary.get("Forward_PE"), summary.get("Revenue_Growth_TTM_YoY"), summary.get("Revenue_Latest_FY")]
        summary["Status"] = "OK" if any(pd.notna(x) for x in useful) else "Limited data"
    except Exception as exc:
        summary["Status"] = f"Error: {type(exc).__name__}"
    return summary, annual_rows


def main():
    tickers = []
    for line in (ROOT / "tickers.txt").read_text().splitlines():
        t = line.strip().upper()
        if t and t not in tickers:
            tickers.append(t)

    summaries = []
    annual = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_one, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            s, a = fut.result()
            summaries.append(s)
            annual.extend(a)
            if i % 50 == 0:
                print(f"Completed {i}/{len(tickers)}", flush=True)
                time.sleep(1)

    sdf = pd.DataFrame(summaries).sort_values("Ticker")
    adf = pd.DataFrame(annual)
    if not adf.empty:
        adf["Fiscal_Year_End"] = pd.to_datetime(adf["Fiscal_Year_End"])
        adf = adf.sort_values(["Ticker", "Fiscal_Year_End"])
        adf["Revenue_Growth_YoY"] = adf.groupby("Ticker")["Revenue"].pct_change()
        adf["Net_Income_Growth_YoY"] = adf.groupby("Ticker")["Net_Income"].pct_change()
    sdf.to_csv(OUT / "fundamentals_summary.csv", index=False)
    adf.to_csv(OUT / "annual_financials.csv", index=False)
    print(sdf["Status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
