import os
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "models", "finbert-finetuned")
NEWS_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sp500_news.csv")
SCORED_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "news_scored.csv")


def run_inference_on_news():
    os.makedirs(os.path.dirname(SCORED_PATH), exist_ok=True)

    df = pd.read_csv(NEWS_PATH)
    print(f"Scoring {len(df)} headlines across {df['ticker'].nunique()} tickers...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    all_preds, all_confidence, all_probs = [], [], []

    batch_size = 32
    for i in range(0, len(df), batch_size):
        batch  = df["title"].iloc[i:i + batch_size].tolist()
        inputs = tokenizer(
            batch,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits

        probs      = softmax(logits, dim=-1).numpy()
        preds      = np.argmax(probs, axis=-1)
        confidence = probs[np.arange(len(preds)), preds]

        all_preds.extend(preds)
        all_confidence.extend(confidence)
        all_probs.extend(probs)

    df["pred_label"]    = all_preds
    df["confidence"]    = all_confidence
    df["prob_negative"] = [p[0] for p in all_probs]
    df["prob_neutral"]  = [p[1] for p in all_probs]
    df["prob_positive"] = [p[2] for p in all_probs]

    df.to_csv(SCORED_PATH, index=False)
    print(f"Saved → {SCORED_PATH}")

    # ── Summary ────────────────────────────────────────────────────────
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    print(f"\nPredicted label distribution:")
    for label, name in label_map.items():
        count = (df["pred_label"] == label).sum()
        pct   = count / len(df) * 100
        print(f"  {name:10s}: {count:4d}  ({pct:.1f}%)")

    high_conf_pos = (df["pred_label"] == 2) & (df["confidence"] > 0.9)
    print(f"\nHigh-confidence positives (conf > 0.9): {high_conf_pos.sum()}")

    print(f"\nSample high-confidence positives:")
    for _, row in df[high_conf_pos].head(8).iterrows():
        print(f"  [{row['confidence']:.3f}] {row['ticker']:5s} | {row['title'][:70]}")

    print(f"\nSample high-confidence negatives:")
    high_conf_neg = (df["pred_label"] == 0) & (df["confidence"] > 0.9)
    for _, row in df[high_conf_neg].head(5).iterrows():
        print(f"  [{row['confidence']:.3f}] {row['ticker']:5s} | {row['title'][:70]}")

    return df


if __name__ == "__main__":
    run_inference_on_news()