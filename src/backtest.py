import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

ALIGNED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "aligned.csv")
OUT_DIR      = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def run_backtest(confidence_threshold: float = 0.9):
    df = pd.read_csv(ALIGNED_PATH)
    print(f"Total headlines: {len(df)}")
    print(f"Confidence threshold: {confidence_threshold}")

    # ── 1. Define the signal ───────────────────────────────────────────
    # Signal = 1 if model predicts positive with high confidence
    df["signal"] = (
        (df["pred_label"] == 2) & (df["confidence"] >= confidence_threshold)
    ).astype(int)

    # Ground truth = 1 if the stock had positive average next-day return
    df["actual_positive"] = (df["avg_next1d_return"] > 0).astype(int)

    n_signals = df["signal"].sum()
    print(f"\nSignal headlines (high-conf positive): {n_signals}")
    print(f"Non-signal headlines: {len(df) - n_signals}")

    # ── 2. Precision / Recall ─────────────────────────────────────────
    precision = precision_score(df["actual_positive"], df["signal"], zero_division=0)
    recall    = recall_score(df["actual_positive"], df["signal"], zero_division=0)
    f1        = f1_score(df["actual_positive"], df["signal"], zero_division=0)

    print(f"\nPrecision : {precision:.4f}  (of headlines flagged positive, how many stocks had positive returns)")
    print(f"Recall    : {recall:.4f}  (of all positive-return stocks, how many did we flag)")
    print(f"F1        : {f1:.4f}")

    # ── 3. Random baseline ────────────────────────────────────────────
    np.random.seed(42)
    n_trials = 1000
    random_precisions = []
    for _ in range(n_trials):
        random_signal = np.random.binomial(1, df["signal"].mean(), size=len(df))
        rp = precision_score(df["actual_positive"], random_signal, zero_division=0)
        random_precisions.append(rp)

    baseline_precision = np.mean(random_precisions)
    print(f"\nRandom baseline precision (mean over {n_trials} trials): {baseline_precision:.4f}")
    print(f"Signal lift over random: {(precision - baseline_precision):+.4f}")

    # ── 4. Sharpe-like signal quality metric ──────────────────────────
    # For headlines where we fired the signal, what were the returns?
    signal_returns    = df[df["signal"] == 1]["avg_next1d_return"]
    nonsignal_returns = df[df["signal"] == 0]["avg_next1d_return"]

    signal_sharpe = (
        signal_returns.mean() / signal_returns.std() * np.sqrt(252)
        if signal_returns.std() > 0 else 0
    )
    nonsignal_sharpe = (
        nonsignal_returns.mean() / nonsignal_returns.std() * np.sqrt(252)
        if nonsignal_returns.std() > 0 else 0
    )

    print(f"\nSharpe-like metric:")
    print(f"  Signal portfolio    : {signal_sharpe:+.4f}")
    print(f"  Non-signal portfolio: {nonsignal_sharpe:+.4f}")
    print(f"  Difference          : {(signal_sharpe - nonsignal_sharpe):+.4f}")

    # ── 5. Return comparison by sentiment ─────────────────────────────
    print(f"\nReturn breakdown by sentiment group:")
    for label, name in [(0, "negative"), (1, "neutral"), (2, "positive")]:
        subset = df[df["pred_label"] == label]["avg_next1d_return"]
        print(f"  {name:10s}: mean={subset.mean():+.6f}  std={subset.std():.6f}  n={len(subset)}")

    # ── 6. Plots ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Return distribution by sentiment
    ax = axes[0]
    for label, name, color in [(0, "negative", "#e74c3c"),
                                (1, "neutral",  "#95a5a6"),
                                (2, "positive", "#2ecc71")]:
        subset = df[df["pred_label"] == label]["avg_next1d_return"]
        ax.hist(subset, bins=20, alpha=0.6, label=name, color=color)
    ax.set_title("Return distribution by sentiment")
    ax.set_xlabel("Avg next-day return")
    ax.set_ylabel("Count")
    ax.legend()

    # Plot 2: Signal vs non-signal returns
    ax = axes[1]
    ax.bar(["Signal\n(high-conf pos)", "Non-signal"],
           [signal_returns.mean(), nonsignal_returns.mean()],
           color=["#2ecc71", "#95a5a6"])
    ax.set_title("Avg return: signal vs non-signal")
    ax.set_ylabel("Avg next-day return")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # Plot 3: Random baseline vs actual precision
    ax = axes[2]
    ax.hist(random_precisions, bins=30, color="#3498db", alpha=0.7, label="Random baseline")
    ax.axvline(precision, color="#e74c3c", linewidth=2, label=f"Signal precision ({precision:.3f})")
    ax.axvline(baseline_precision, color="#2ecc71", linewidth=2,
               linestyle="--", label=f"Baseline mean ({baseline_precision:.3f})")
    ax.set_title("Precision vs random baseline")
    ax.set_xlabel("Precision")
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "backtest_results.png")
    plt.savefig(out_path, dpi=120)
    plt.show()
    print(f"\nPlot saved → {out_path}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "baseline_precision": baseline_precision,
        "signal_sharpe": signal_sharpe,
        "nonsignal_sharpe": nonsignal_sharpe,
    }


if __name__ == "__main__":
    results = run_backtest(confidence_threshold=0.9)