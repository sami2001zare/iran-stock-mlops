import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess(raw_path="data/raw/stock_data.csv", 
               processed_path="data/processed/clean_data.csv"):
    df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["date"])
    
    for col in ["open", "high", "low", "close", "volume"]:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] > mean - 5*std) & (df[col] < mean + 5*std)]
    
    df = df.dropna()
    
    df = df.sort_values(["symbol", "date"])
    
    df.to_csv(processed_path, index=False)
    print(f"Preprocessed data saved to {processed_path}. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    preprocess()