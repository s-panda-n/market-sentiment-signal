# market-sentiment-signal

News Sentiment to Price Movement — FinBERT fine-tuning + backtest

## What this project does

Fine-tunes FinBERT on the `financial_phrasebank` dataset to classify
financial news headlines as positive / neutral / negative, then tests
whether high-confidence positive sentiment predicts positive next-day
returns for S&P 500 stocks.

## Pipeline

financial_phrasebank  →  fine-tune FinBERT  →  inference on news headlines
↓
S&P 500 price data   →  compute returns    →  join on ticker  →  backtest

## Results

| Metric | Value |
|--------|-------|
| FinBERT test accuracy (phrasebank) | 99% |
| High-confidence positive headlines | 209 / 2009 |
| Signal precision | 0.971 |
| Random baseline precision | 0.962 |
| Signal lift | +0.0095 |
| Signal Sharpe-like metric | 21.23 |
| Non-signal Sharpe-like metric | 23.03 |

## Honest limitations

**No date-based join.** The news headlines (March–April 2026) do not
overlap with the price dataset (up to Feb 2026). Returns are joined on
ticker only, using each stock's 5-year average next-day return. This
means every headline for the same ticker gets the same return value,
which inflates precision and makes the backtest less meaningful than
a true event-study design would be.

**Bull market base rate.** Almost all S&P 500 stocks had positive
average returns over 2021–2026, so the random baseline precision is
already ~96%. Any signal operating on this data will appear to have
high precision by default.

**Phrasebank is European.** The `sentences_allagree` split is dominated
by Finnish/European companies not in the S&P 500, so the model was
fine-tuned on out-of-domain company names. Match rate to S&P 500
tickers was only ~5%.

## What a stronger version would look like

- Use a news API with full historical coverage (e.g. Polygon.io news,
  Bloomberg, or Refinitiv) to get dated headlines per ticker
- Join sentiment scores to same-day or next-day returns by date + ticker
- Use an event-study design: measure abnormal returns around
  high-sentiment headline dates vs. the stock's historical baseline
- Evaluate on out-of-sample time periods

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python src/data_collection.py   # download price data
python src/fetch_news.py        # fetch headlines via NewsAPI
python src/inference.py         # score headlines with FinBERT
python src/price_alignment.py   # join sentiment with returns
python src/backtest.py          # run backtest and generate plots
```

## Stack

- `transformers` — FinBERT fine-tuning and inference
- `financial_phrasebank` — labeled financial sentences (HuggingFace)
- `pandas-datareader` / Kaggle CSV — S&P 500 price data
- `NewsAPI` — news headlines
- `rapidfuzz` — fuzzy company name matching
- `scikit-learn` — precision / recall metrics