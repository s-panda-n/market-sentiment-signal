import os
import requests
import pandas as pd
from datasets import load_dataset
from datetime import datetime, timedelta
from io import StringIO

# ── Paths ──────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


# ── 1. Financial PhraseBank ────────────────────────────────────────────
def load_phrasebank():
    """
    Loads the financial_phrasebank dataset (sentences_allagree split).
    Labels: 0 = negative, 1 = neutral, 2 = positive
    Returns a pandas DataFrame with columns: [sentence, label]
    """
    print("Downloading financial_phrasebank...")
    ds = load_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)

    df = ds["train"].to_pandas()
    df.columns = ["sentence", "label"]

    out_path = os.path.join(RAW_DIR, "phrasebank.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")
    return df


# ── 2. S&P 500 tickers ────────────────────────────────────────────────
def get_sp500_tickers():
    """
    Fetches S&P 500 tickers from Wikipedia.
    Falls back to a hardcoded top-50 list if that fails.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        table = pd.read_html(StringIO(resp.text))[0]
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"Fetched {len(tickers)} tickers from Wikipedia.")
        return tickers
    except Exception as e:
        print(f"Wikipedia fetch failed ({e}), using fallback ticker list.")
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "UNH", "JPM", "V", "XOM", "LLY", "JNJ", "MA", "AVGO", "PG", "HD",
            "CVX", "MRK", "ABBV", "COST", "PEP", "ADBE", "WMT", "BAC", "KO",
            "MCD", "CRM", "ACN", "TMO", "CSCO", "ABT", "NFLX", "LIN", "DHR",
            "ORCL", "AMD", "TXN", "NEE", "PM", "UPS", "MS", "RTX", "INTC",
            "AMGN", "QCOM", "INTU", "LOW", "SPGI"
        ]


# ── 3. Load price data from Kaggle CSV ────────────────────────────────
def load_price_data_from_csv(csv_path: str, period_years: int = 2) -> dict:
    """
    Loads S&P 500 OHLCV data from the Kaggle sp500_stocks.csv file.
    Columns: Date, Symbol, Adj Close, Close, High, Low, Open, Volume

    Filters to the last `period_years` years and drops rows where
    Close is NaN (sparse early data like MMM 2010).

    Returns a dict of {ticker: DataFrame}, index = Date.
    """
    print(f"Loading price data from {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=["Date"])

    # Filter to recent window
    cutoff = datetime.today() - timedelta(days=365 * period_years)
    df = df[df["Date"] >= cutoff]

    # Drop rows with no Close price
    df = df.dropna(subset=["Close"])

    if df.empty:
        print("ERROR: CSV is empty after filtering. Check the file and period_years.")
        return {}

    price_data = {}
    for ticker, group in df.groupby("Symbol"):
        group = group.sort_values("Date").set_index("Date")
        group.index = pd.to_datetime(group.index).tz_localize(None)
        group["ticker"] = ticker
        price_data[ticker] = group

    print(f"Loaded {len(price_data)} tickers after filtering.")
    return price_data


# ── 4. Compute daily returns ──────────────────────────────────────────
def compute_returns(price_data: dict) -> pd.DataFrame:
    """
    Computes same-day and next-day close-to-close returns for all tickers.
    Returns a long-format DataFrame: [date, ticker, close, return_1d, return_next_1d]
    """
    if not price_data:
        print("ERROR: No price data to compute returns from.")
        return pd.DataFrame()

    frames = []
    for ticker, df in price_data.items():
        close_col = [c for c in df.columns if c.lower() == "close"][0]
        tmp = df[[close_col]].copy()
        tmp.columns = ["close"]
        tmp["ticker"] = ticker
        tmp["return_1d"] = tmp["close"].pct_change()
        tmp["return_next_1d"] = tmp["return_1d"].shift(-1)
        tmp.index.name = "date"
        frames.append(tmp.reset_index())

    returns_df = pd.concat(frames, ignore_index=True)
    returns_df.dropna(subset=["return_1d", "return_next_1d"], inplace=True)

    out_path = os.path.join(RAW_DIR, "returns.csv")
    returns_df.to_csv(out_path, index=False)
    print(f"Saved returns → {out_path}  ({len(returns_df)} rows)")
    return returns_df


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Sentiment dataset
    phrasebank_df = load_phrasebank()
    print(phrasebank_df["label"].value_counts(), "\n")

    # 2. Price data from Kaggle CSV
    csv_path = os.path.join(RAW_DIR, "sp500_stocks.csv")
    price_data = load_price_data_from_csv(csv_path, period_years=2)

    # 3. Returns
    returns_df = compute_returns(price_data)
    print(returns_df.head())
    print(f"\nDate range: {returns_df['date'].min()} → {returns_df['date'].max()}")
    print(f"Tickers:    {returns_df['ticker'].nunique()}")
    print(f"Total rows: {len(returns_df)}")