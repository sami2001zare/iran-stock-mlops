import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from datetime import datetime

def train_stateless(X_train, y_train, X_val, y_val, params=None):
    if params is None:
        params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
    model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

def train_stateful_sliding_window(X_all, y_all, window_size=500, step=1):
    n = len(X_all)
    if n <= window_size:
        return train_stateless(X_all, y_all, None, None)
    X_recent = X_all.iloc[-window_size:]
    y_recent = y_all.iloc[-window_size:]
    return train_stateless(X_recent, y_recent, None, None)

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = joblib.load("data/processed/splits.pkl")
    
    USE_STATEFUL = True   # Set to False for full historical stateless
    
    mlflow.set_experiment("iran-stock-prediction")
    
    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        if USE_STATEFUL:
            X_combined = pd.concat([X_train, X_val], axis=0)
            y_combined = pd.concat([y_train, y_val], axis=0)
            model = train_stateful_sliding_window(X_combined, y_combined, window_size=500)
            mlflow.log_param("training_strategy", "stateful_sliding_window")
            mlflow.log_param("window_size", 500)
        else:
            model = train_stateless(X_train, y_train, X_val, y_val)
            mlflow.log_param("training_strategy", "stateless_full_history")
        
        y_val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred)
        rec = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_precision", prec)
        mlflow.log_metric("val_recall", rec)
        mlflow.log_metric("val_f1", f1)
        
        mlflow.log_params(model.get_params())
        
        joblib.dump(model, "models/latest_model.pkl")
        mlflow.sklearn.log_model(model, "model")
        
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "StockDirectionModel")
        
        print(f"Model trained. Val accuracy: {acc:.4f}")