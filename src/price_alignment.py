import os
import pandas as pd

SCORED_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "news_scored.csv")
RETURNS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "returns.csv")
OUT_PATH     = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "aligned.csv")


def align():
    news_df    = pd.read_csv(SCORED_PATH)
    returns_df = pd.read_csv(RETURNS_PATH, parse_dates=["date"])

    print(f"Headlines scored : {len(news_df)}")
    print(f"Tickers in news  : {news_df['ticker'].nunique()}")
    print(f"Returns rows     : {len(returns_df)}")

    # ── Summarise returns per ticker ───────────────────────────────────
    # No date overlap between news (Mar-Apr 2026) and price data (to Feb 2026),
    # so we join on ticker only and use historical average return as the signal.
    returns_summary = (
        returns_df.groupby("ticker")["return_next_1d"]
        .agg(["mean", "std", "count", "median"])
        .reset_index()
        .rename(columns={
            "mean":   "avg_next1d_return",
            "std":    "std_next1d_return",
            "count":  "n_trading_days",
            "median": "median_next1d_return",
        })
    )

    # ── Join on ticker ─────────────────────────────────────────────────
    aligned_df = news_df.merge(returns_summary, on="ticker", how="inner")

    aligned_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}  ({len(aligned_df)} rows)")

    # ── Summary stats by predicted label ──────────────────────────────
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    print(f"\nAverage next-day return by predicted sentiment:")
    print(f"  {'label':10s}  {'count':>6}  {'avg_return':>12}  {'std':>10}")
    print(f"  {'-'*44}")
    for label, name in label_map.items():
        subset = aligned_df[aligned_df["pred_label"] == label]
        if len(subset) == 0:
            continue
        avg = subset["avg_next1d_return"].mean()
        std = subset["avg_next1d_return"].std()
        print(f"  {name:10s}  {len(subset):>6}  {avg:>+12.6f}  {std:>10.6f}")

    # ── High-confidence positives ──────────────────────────────────────
    high_pos = aligned_df[(aligned_df["pred_label"] == 2) & (aligned_df["confidence"] > 0.9)]
    high_neg = aligned_df[(aligned_df["pred_label"] == 0) & (aligned_df["confidence"] > 0.9)]

    print(f"\nHigh-confidence positives (conf > 0.9): {len(high_pos)}")
    print(f"  Avg return: {high_pos['avg_next1d_return'].mean():+.6f}")

    print(f"\nHigh-confidence negatives (conf > 0.9): {len(high_neg)}")
    print(f"  Avg return: {high_neg['avg_next1d_return'].mean():+.6f}")

    print(f"\nSample high-confidence positives:")
    cols = ["ticker", "confidence", "avg_next1d_return", "title"]
    for _, row in high_pos.head(6).iterrows():
        print(f"  [{row['confidence']:.3f}] {row['ticker']:5s} | return={row['avg_next1d_return']:+.4f} | {row['title'][:60]}")

    return aligned_df


if __name__ == "__main__":
    align()