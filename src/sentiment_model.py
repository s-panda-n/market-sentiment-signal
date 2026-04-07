import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import evaluate
import torch

# ── Config ─────────────────────────────────────────────────────────────
MODEL_NAME  = "ProsusAI/finbert"
NUM_LABELS  = 3
MAX_LENGTH  = 128
EPOCHS      = 5
BATCH_SIZE  = 16
LR          = 2e-5
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "models", "finbert-finetuned")
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "phrasebank.csv")

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Split: 80% train, 10% val, 10% test
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    return train_df, val_df, test_df

def tokenize(df, tokenizer):
    ds = Dataset.from_pandas(df.reset_index(drop=True))
    def _tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
    return ds.map(_tokenize, batched=True)

def get_class_weights(train_df):
    counts = train_df["label"].value_counts().sort_index()  # [neg, neu, pos]
    total  = len(train_df)
    weights = total / (NUM_LABELS * counts)
    print(f"Class weights: {weights.round(3).tolist()}")
    return torch.tensor(weights.values, dtype=torch.float)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    return acc

def train():
    train_df, val_df, test_df = load_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
        ignore_mismatched_sizes=True,  # reinitializes the classification head
    )

    train_ds = tokenize(train_df, tokenizer)
    val_ds   = tokenize(val_df,   tokenizer)
    test_ds  = tokenize(test_df,  tokenizer)

    class_weights = get_class_weights(train_df)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=20,
        seed=42,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # Evaluate on held-out test set
    print("\nTest set evaluation:")
    preds_output = trainer.predict(test_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    print(classification_report(test_df["label"], preds, target_names=["negative", "neutral", "positive"]))

    # Save model and tokenizer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    train()