import os
import time
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta

API_KEY  = "774ac7c2721a4bdca0c39065520d86ed"
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sp500_news.csv")

TICKERS_AND_NAMES = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "GOOGL": "Google OR Alphabet",
    "AMZN":  "Amazon",
    "NVDA":  "Nvidia",
    "META":  "Meta OR Facebook",
    "TSLA":  "Tesla",
    "JPM":   "JPMorgan",
    "BAC":   "Bank of America",
    "GS":    "Goldman Sachs",
    "MS":    "Morgan Stanley",
    "V":     "Visa",
    "MA":    "Mastercard",
    "JNJ":   "Johnson Johnson",
    "PFE":   "Pfizer",
    "MRK":   "Merck",
    "XOM":   "ExxonMobil",
    "CVX":   "Chevron",
    "WMT":   "Walmart",
    "NFLX":  "Netflix",
    "AMD":   "AMD semiconductor",
    "INTC":  "Intel",
    "COST":  "Costco",
    "ADBE":  "Adobe",
    "CRM":   "Salesforce",
}

def fetch_all():
    api = NewsApiClient(api_key=API_KEY)

    # Free tier: 30 days back from today
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=29)

    all_rows = []

    for ticker, query in TICKERS_AND_NAMES.items():
        try:
            resp = api.get_everything(
                q=query,
                from_param=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                language="en",
                sort_by="publishedAt",
                page_size=100,
            )
            articles = resp.get("articles", [])
            for a in articles:
                all_rows.append({
                    "ticker":    ticker,
                    "title":     a.get("title", "") or "",
                    "published": a.get("publishedAt", ""),
                    "source":    a.get("source", {}).get("name", ""),
                })
            print(f"  {ticker:6s}: {len(articles)} articles")
            time.sleep(0.3)

        except Exception as e:
            print(f"  {ticker:6s}: FAILED — {e}")

    if not all_rows:
        print("ERROR: No articles fetched. Check your API key.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df[df["title"].str.len() > 10].drop_duplicates(subset=["title"])
    df["date"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["date"])

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} headlines → {OUT_PATH}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    return df

if __name__ == "__main__":
    fetch_all()