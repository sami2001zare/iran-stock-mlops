# pipeline/ingest.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(symbols=["IKCO1", "FOLD1", "KHOD1"], 
                            start_date="2020-01-01", 
                            end_date="2023-12-31"):
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # business days
    records = []
    for symbol in symbols:
        base_price = np.random.uniform(5000, 20000)
        returns = np.random.normal(0.001, 0.02, len(date_range))
        prices = base_price * np.exp(np.cumsum(returns))
        for i, date in enumerate(date_range):
            open_price = prices[i]
            close_price = open_price * (1 + np.random.normal(0.001, 0.02))
            high = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(10000, 1000000)
            records.append([date.date(), symbol, open_price, high, low, close_price, volume])
    df = pd.DataFrame(records, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    df.to_csv("data/raw/stock_data.csv", index=False)
    print(f"Saved raw data to data/raw/stock_data.csv ({len(df)} rows)")
    return df

if __name__ == "__main__":
    generate_synthetic_data()