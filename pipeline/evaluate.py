import mlflow
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def evaluate():
    # Load test splits
    X_train, y_train, X_val, y_val, X_test, y_test = joblib.load("data/processed/splits.pkl")
    
    # Load model from registry (latest version)
    client = mlflow.tracking.MlflowClient()
    model_name = "StockDirectionModel"
    latest_version = client.get_latest_versions(model_name, stages=["None"])[-1].version
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    
    y_pred = model.predict(X_test)
    print("Test Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate()