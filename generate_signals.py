#!/usr/bin/env python3
"""
generate_signals.py  -  Step 5 of the "LLMs for Sentiment Analysis in Trading"
pipeline: turn FinBERT scores into buy/sell signals and test them on SPY.

From the course
  * thresholds come from quantiles of the sentiment score distribution;
  * a score above the upper threshold is a buy signal (below the lower one, sell);
  * decisions every 30 seconds on 30-second SPY bars;
  * each bar is usable only 15 seconds after it ends (embargo, done in step 2).

Choices not visible in the course material (adjustable)
  * thresholds: 80th and 20th percentiles (--upper 0.8 --lower 0.2);
  * between the thresholds the position is flat (no position);
  * a position is held for one 30-second step, then the next signal decides
    again; a repeated signal keeps the position, so costs are paid only when the
    position changes;
  * trades happen at the first 30-second SPY bar close at or after the moment a
    bar becomes usable (the bar's "available" time, at :15 or :45, is rounded up
    to the next :00 or :30), so a signal never uses a price from before it existed;
  * costs: 0.5 basis points of SPY's price for every unit of position change.
  * hit rate: share of long/short steps where SPY moved in the signal's direction,
    counting only steps where SPY's price changed.

Threshold modes
  insample   quantiles over all five press conferences, as in the course. This
             uses scores that were not yet known at the time (look-ahead).
  expanding  quantiles over earlier press conferences only; the first one gets no
             trades. This is what you could actually have traded.

Input   bars/bar_sentiment.csv            (score_sentiment.py)
        optional SPY intraday prices:  CSV with a timestamp and a price column,
        any frequency (30-second, 1-minute, ticks); resampled to 30 seconds.
        Timestamps are taken as US Eastern time unless they carry a time zone or
        you pass --spy-tz (e.g. UTC).
Output  signals/signals.csv               one row per bar: score, thresholds, signal
        signals/backtest_trades.csv       with --spy: position and return per step
        signals/backtest_summary.csv      with --spy: results per press conference

Usage
  python generate_signals.py                                # signals only
  python generate_signals.py --spy spy_30s.csv              # signals + backtest
  python generate_signals.py --spy spy_1min.csv --spy-tz UTC --mode expanding
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STEP = pd.Timedelta(seconds=30)


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
def thresholds(df: pd.DataFrame, upper: float, lower: float, mode: str) -> pd.DataFrame:
    """Upper and lower score thresholds for every bar."""
    df = df.copy()
    if mode == "insample":
        df["thr_upper"] = df["score"].quantile(upper)
        df["thr_lower"] = df["score"].quantile(lower)
        return df
    meetings = sorted(df["meeting_date"].unique())
    for m in meetings:
        past = df.loc[df["meeting_date"] < m, "score"]
        sel = df["meeting_date"] == m
        df.loc[sel, "thr_upper"] = past.quantile(upper) if len(past) else np.nan
        df.loc[sel, "thr_lower"] = past.quantile(lower) if len(past) else np.nan
    return df


def make_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sig = np.where(df["score"] > df["thr_upper"], 1,
                   np.where(df["score"] < df["thr_lower"], -1, 0))
    sig = np.where(df["thr_upper"].isna(), 0, sig)
    df["signal"] = sig.astype(int)
    df["action"] = df["signal"].map({1: "BUY", -1: "SELL", 0: "FLAT"})
    # trade time: first 30-second mark at or after the bar becomes usable
    avail = pd.to_datetime(df["available_et"])
    df["trade_et"] = avail.dt.ceil("30s")
    df["exit_et"] = df["trade_et"] + STEP
    return df


# --------------------------------------------------------------------------
# SPY prices
# --------------------------------------------------------------------------
def load_spy(path: str, tz: str | None) -> pd.Series:
    raw = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in raw.columns}
    tcol = next((cols[c] for c in ("timestamp", "datetime", "date_time", "time", "date", "t")
                 if c in cols), None)
    pcol = next((cols[c] for c in ("close", "price", "last", "c", "adj close") if c in cols), None)
    if tcol is None or pcol is None:
        sys.exit(f"SPY file needs a timestamp column and a close/price column; "
                 f"found {list(raw.columns)}")
    ts = pd.to_datetime(raw[tcol], format="mixed", utc=False)
    if ts.dt.tz is None and tz:
        ts = ts.dt.tz_localize(tz)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    px = pd.Series(pd.to_numeric(raw[pcol], errors="coerce").to_numpy(), index=ts).dropna()
    return px.sort_index()


def price_at(px: pd.Series, times: pd.Series) -> np.ndarray:
    """Last price at or before each time (a bar's close is stamped at its end)."""
    idx = px.index.searchsorted(times.to_numpy(), side="right") - 1
    out = np.full(len(times), np.nan)
    ok = idx >= 0
    out[ok] = px.to_numpy()[idx[ok]]
    # refuse prices staler than 2 minutes (e.g. a gap in the data)
    stale = np.zeros(len(times), bool)
    stale[ok] = (times.to_numpy()[ok] - px.index.to_numpy()[idx[ok]]) > np.timedelta64(120, "s")
    out[stale] = np.nan
    return out


def frequency_note(px: pd.Series, days) -> str:
    d = px[px.index.normalize().isin(pd.to_datetime(list(days)))]
    if len(d) < 3:
        return "no SPY prices on the press conference days"
    step = pd.Series(d.index).diff().dt.total_seconds().median()
    note = f"median spacing of SPY prices on those days: {step:.0f} s"
    if step > 30:
        note += (" - coarser than the course's 30-second bars, so some 30-second steps "
                 "show no price change; results are only an approximation")
    return note


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def backtest(sig: pd.DataFrame, px: pd.Series, cost_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = sig[["meeting_date", "bar", "score", "signal", "trade_et", "exit_et"]].copy()
    t["px_entry"] = price_at(px, t["trade_et"])
    t["px_exit"] = price_at(px, t["exit_et"])
    t["spy_ret"] = t["px_exit"] / t["px_entry"] - 1
    t["position"] = t["signal"]
    prev = t.groupby("meeting_date")["position"].shift(1).fillna(0)
    t["turnover"] = (t["position"] - prev).abs()
    # close the last position of each press conference
    last = t.groupby("meeting_date").cumcount(ascending=False) == 0
    t.loc[last, "turnover"] += t.loc[last, "position"].abs()
    t["gross_ret"] = t["position"] * t["spy_ret"]
    t["cost"] = t["turnover"] * cost_bps / 1e4
    t["net_ret"] = t["gross_ret"] - t["cost"]
    valid = t["spy_ret"].notna()

    rows = []
    for m, g in list(t[valid].groupby("meeting_date")) + [("ALL", t[valid])]:
        # hit rate over positions where SPY actually moved (with prices coarser than
        # 30 s, many steps show no change and would otherwise count as misses)
        traded = g[(g["position"] != 0) & (g["spy_ret"] != 0)]
        hits = (np.sign(traded["spy_ret"]) == traded["position"]).mean() if len(traded) else np.nan
        bh = (1 + g["spy_ret"]).prod() - 1 if m != "ALL" else \
            g.groupby("meeting_date")["spy_ret"].apply(lambda r: (1 + r).prod() - 1).mean()
        rows.append({"meeting_date": m, "steps": len(g),
                     "steps_long": int((g["position"] > 0).sum()),
                     "steps_short": int((g["position"] < 0).sum()),
                     "trades": int((g["turnover"] > 0).sum()),
                     "hit_rate": round(hits, 3) if hits == hits else np.nan,
                     "strategy_gross_bp": round(((1 + g["gross_ret"]).prod() - 1) * 1e4, 2),
                     "strategy_net_bp": round(((1 + g["net_ret"]).prod() - 1) * 1e4, 2),
                     "spy_buy_hold_bp": round(bh * 1e4, 2)})
    summary = pd.DataFrame(rows)
    summary.loc[summary["meeting_date"] == "ALL", "spy_buy_hold_bp"] = np.nan
    return t, summary


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Sentiment signals and SPY backtest")
    ap.add_argument("--bars", default="bars")
    ap.add_argument("--out", default="signals")
    ap.add_argument("--upper", type=float, default=0.8, help="upper quantile (default 0.8)")
    ap.add_argument("--lower", type=float, default=0.2, help="lower quantile (default 0.2)")
    ap.add_argument("--mode", default="insample", choices=["insample", "expanding"])
    ap.add_argument("--spy", help="CSV of intraday SPY prices (timestamp, close)")
    ap.add_argument("--spy-tz", help="time zone of naive SPY timestamps (default: US Eastern)")
    ap.add_argument("--cost-bps", type=float, default=0.5,
                    help="cost per unit of position change, in basis points (default 0.5)")
    args = ap.parse_args()

    src = Path(args.bars) / "bar_sentiment.csv"
    if not src.exists():
        sys.exit(f"{src} not found. Run score_sentiment.py first.")
    df = pd.read_csv(src)
    df = df[pd.to_numeric(df["score"], errors="coerce").notna()].copy()
    df["score"] = df["score"].astype(float)
    df = df.sort_values(["meeting_date", "bar"]).reset_index(drop=True)

    sig = make_signals(thresholds(df, args.upper, args.lower, args.mode))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = ["meeting_date", "video_id", "bar", "start_et", "end_et", "available_et",
            "trade_et", "exit_et", "score", "label", "thr_lower", "thr_upper", "signal",
            "action", "text"]
    sig[keep].to_csv(out / "signals.csv", index=False)

    print(f"Thresholds ({args.mode}): buy above the {args.upper:.0%} quantile, "
          f"sell below the {args.lower:.0%} quantile")
    if args.mode == "insample":
        print(f"  upper {sig['thr_upper'].iloc[0]:.3f}, lower {sig['thr_lower'].iloc[0]:.3f}"
              "  (uses all five meetings: look-ahead, as in the course)")
    counts = sig.groupby("meeting_date")["action"].value_counts().unstack(fill_value=0)
    counts = counts.reindex(columns=["BUY", "SELL", "FLAT"], fill_value=0)
    thr = sig.groupby("meeting_date")[["thr_lower", "thr_upper"]].first().round(3)
    print("\n" + counts.join(thr).to_string())
    print(f"\nSignals written to {out / 'signals.csv'}")

    if not args.spy:
        print("\nNo SPY prices given (--spy), so no backtest yet.")
        return

    px = load_spy(args.spy, args.spy_tz)
    print(f"\nSPY prices: {len(px):,} rows, {px.index.min()} to {px.index.max()} (ET)")
    print("  " + frequency_note(px, sig["meeting_date"].unique()))
    trades, summary = backtest(sig, px, args.cost_bps)
    trades.to_csv(out / "backtest_trades.csv", index=False)
    summary.to_csv(out / "backtest_summary.csv", index=False)
    n_miss = int(trades["spy_ret"].isna().sum())
    if n_miss:
        print(f"  {n_miss} of {len(trades)} steps have no SPY price and are left out")
    print(f"\nBacktest (returns in basis points, costs {args.cost_bps} bp per unit traded):")
    print(summary.to_string(index=False))
    print(f"\nWritten to {out / 'backtest_trades.csv'} and {out / 'backtest_summary.csv'}")


if __name__ == "__main__":
    main()
