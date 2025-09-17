"""Run monthly backtests and enforce win-rate/ROI constraints."""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class MonthResult:
    period: str
    trades: int
    winrate: float
    roi: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling monthly backtests")
    parser.add_argument("--strategy", default="EnsembleMetaStrategy", help="Strategy class to backtest")
    parser.add_argument("--config", type=Path, default=None, help="Optional freqtrade configuration file")
    parser.add_argument("--start", default="2024-01", help="First month (YYYY-MM)")
    parser.add_argument("--end", default="2025-06", help="Last month (YYYY-MM)")
    parser.add_argument("--min-winrate", type=float, default=0.80, help="Minimum acceptable win-rate per month")
    parser.add_argument("--output-dir", type=Path, default=Path("user_data/backtest_results/monthly"))
    parser.add_argument("--trade-export", default="trades", choices=["trades"], help="Type of export to generate")
    parser.add_argument(
        "--timerange-buffer",
        type=int,
        default=0,
        help="Extend timerange by N days on each side to allow for warmup",
    )
    return parser.parse_args()


def build_timerange(period: pd.Period, buffer_days: int) -> str:
    start = period.start_time - pd.Timedelta(days=buffer_days)
    end = period.end_time + pd.Timedelta(days=buffer_days)
    return f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"


def run_backtest(period: pd.Period, args: argparse.Namespace, export_path: Path) -> MonthResult:
    timerange = build_timerange(period, args.timerange_buffer)
    cmd = [
        "freqtrade",
        "backtesting",
        "--strategy",
        args.strategy,
        "--timerange",
        timerange,
        "--export",
        args.trade_export,
        "--export-filename",
        str(export_path),
    ]
    if args.config:
        cmd.extend(["--config", str(args.config)])

    print(f"Running backtest for {period} ({timerange})")
    subprocess.run(cmd, check=True)

    if not export_path.exists():
        raise RuntimeError(f"Expected export not found: {export_path}")

    trades = pd.read_json(export_path)
    if trades.empty:
        return MonthResult(period=str(period), trades=0, winrate=0.0, roi=0.0)

    trades["close_date"] = pd.to_datetime(trades["close_date"], utc=True, errors="coerce")
    trades["profit_ratio"] = trades["profit_ratio"].astype(float)

    month_mask = trades["close_date"].dt.to_period("M") == period
    trades_in_month = trades.loc[month_mask]
    if trades_in_month.empty:
        trades_in_month = trades

    winrate = float((trades_in_month["profit_ratio"] > 0).mean())
    roi = float(trades_in_month["profit_ratio"].sum())

    return MonthResult(period=str(period), trades=len(trades_in_month), winrate=winrate, roi=roi)


def main() -> None:
    args = parse_arguments()
    months = pd.period_range(args.start, args.end, freq="M")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[MonthResult] = []
    for period in months:
        export_path = args.output_dir / f"trades_{period}.json"
        result = run_backtest(period, args, export_path)
        print(
            f"{period}: trades={result.trades} winrate={result.winrate:.3f} roi={result.roi:.4f}"
        )
        if result.trades == 0:
            raise RuntimeError(f"No trades generated for period {period}")
        if result.winrate < args.min_winrate:
            raise RuntimeError(
                f"Win-rate {result.winrate:.3f} below threshold {args.min_winrate:.2f} for {period}"
            )
        if result.roi <= 0:
            raise RuntimeError(f"Non-positive ROI {result.roi:.4f} detected for {period}")
        results.append(result)

    summary_path = args.output_dir / "monthly_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump([result.__dict__ for result in results], fh, indent=2)

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
