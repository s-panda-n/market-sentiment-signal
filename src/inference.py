import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "finbert-finetuned")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "phrasebank.csv")
OUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "phrasebank_scored.csv")

def run_inference():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print(f"Scoring {len(df)} sentences...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    all_preds      = []
    all_confidence = []
    all_probs      = []

    batch_size = 32
    for i in range(0, len(df), batch_size):
        batch = df["sentence"].iloc[i:i+batch_size].tolist()
        inputs = tokenizer(batch, truncation=True, max_length=128,
                           padding=True, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        probs      = softmax(logits, dim=-1).numpy()
        preds      = np.argmax(probs, axis=-1)
        confidence = probs[np.arange(len(preds)), preds]

        all_preds.extend(preds)
        all_confidence.extend(confidence)
        all_probs.extend(probs)

    df["pred_label"]      = all_preds
    df["confidence"]      = all_confidence
    df["prob_negative"]   = [p[0] for p in all_probs]
    df["prob_neutral"]    = [p[1] for p in all_probs]
    df["prob_positive"]   = [p[2] for p in all_probs]

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved → {OUT_PATH}")

    # Quick sanity check
    print(f"\nPredicted label distribution:")
    print(df["pred_label"].value_counts())
    print(f"\nHigh-confidence positives (conf > 0.9): {((df['pred_label'] == 2) & (df['confidence'] > 0.9)).sum()}")
    print(f"\nSample high-confidence positives:")
    sample = df[(df["pred_label"] == 2) & (df["confidence"] > 0.9)].head(3)
    for _, row in sample.iterrows():
        print(f"  [{row['confidence']:.3f}] {row['sentence'][:80]}")

if __name__ == "__main__":
    run_inference()