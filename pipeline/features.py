import pandas as pd
import numpy as np

def add_features(df):
    df = df.copy()
    symbols = df["symbol"].unique()
    all_dfs = []
    for sym in symbols:
        sub = df[df["symbol"] == sym].sort_values("date")
        
        sub["ma5"] = sub["close"].rolling(5).mean()
        sub["ma10"] = sub["close"].rolling(10).mean()
        sub["ma20"] = sub["close"].rolling(20).mean()
        
        delta = sub["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        sub["rsi"] = 100 - (100 / (1 + rs))
        
        exp1 = sub["close"].ewm(span=12, adjust=False).mean()
        exp2 = sub["close"].ewm(span=26, adjust=False).mean()
        sub["macd"] = exp1 - exp2
        sub["macd_signal"] = sub["macd"].ewm(span=9, adjust=False).mean()
        
        sub["volume_ma5"] = sub["volume"].rolling(5).mean()
        sub["volume_ratio"] = sub["volume"] / sub["volume_ma5"]
        
        sub["target"] = (sub["close"].shift(-1) > sub["close"]).astype(int)
        
        sub = sub.dropna().reset_index(drop=True)
        all_dfs.append(sub)
    
    final = pd.concat(all_dfs, ignore_index=True)
    return final

def create_train_val_test(df, val_ratio=0.1, test_ratio=0.1):
    df = df.sort_values("date")
    dates = df["date"].unique()
    n = len(dates)
    test_cut = int(n * (1 - test_ratio))
    val_cut = int(n * (1 - test_ratio - val_ratio))
    test_start = dates[test_cut]
    val_start = dates[val_cut]
    
    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    
    feature_cols = ["open", "high", "low", "close", "volume", 
                    "ma5", "ma10", "ma20", "rsi", "macd", "macd_signal", 
                    "volume_ma5", "volume_ratio"]
    X_train = train[feature_cols]
    y_train = train["target"]
    X_val = val[feature_cols]
    y_val = val["target"]
    X_test = test[feature_cols]
    y_test = test["target"]
    
    import joblib
    joblib.dump((X_train, y_train, X_val, y_val, X_test, y_test), "data/processed/splits.pkl")
    print("Saved train/val/test splits to data/processed/splits.pkl")
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    clean_df = pd.read_csv("data/processed/clean_data.csv")
    clean_df["date"] = pd.to_datetime(clean_df["date"])
    featurized = add_features(clean_df)
    featurized.to_csv("data/processed/featurized_data.csv", index=False)
    create_train_val_test(featurized)