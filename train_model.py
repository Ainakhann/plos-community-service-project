"""
Model Training Script for Rehabilitation Hospital PLOS Project.
Loads data, performs feature engineering & preprocessing, compares ML models,
evaluates performance metrics, and saves exported artifacts to models/.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from src.generate_data import generate_patient_dataset
from src.preprocessing import load_and_preprocess_data
from src.evaluation import evaluate_model_performance

MODEL_DIR = "models"
DATA_PATH = "data/patient_data.csv"

def train_and_evaluate():
    print("==================================================")
    print(" PLOS MODEL TRAINING & EVALUATION PIPELINE")
    print("==================================================")
    
    # 1. Dataset Generation if missing
    if not os.path.exists(DATA_PATH):
        print("[i] Dataset not found. Generating synthetic dataset...")
        generate_patient_dataset(num_samples=1000, output_path=DATA_PATH)
    else:
        print(f"[+] Dataset found at '{DATA_PATH}'.")
        
    # 2. Data Preprocessing & Feature Engineering & Train/Test Split
    X_train, X_test, y_train, y_test, preprocessor, df_full = load_and_preprocess_data(
        csv_path=DATA_PATH, test_size=0.2, random_state=42
    )
    
    # 3. Model Comparison & Training
    candidate_models = {
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Linear Regression": LinearRegression()
    }
    
    best_model_name = None
    best_model = None
    best_metrics = None
    lowest_rmse = float("inf")
    
    print("\n--- Model Candidate Comparison ---")
    comparison_summary = {}
    
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model_performance(y_test, y_pred)
        
        rmse = metrics["regression"]["rmse"]
        r2 = metrics["regression"]["r2_score"]
        acc = metrics["classification"]["accuracy"]
        
        print(f"-> {name}:")
        print(f"   RMSE: {rmse:.3f} days | R²: {r2:.4f} | Category Accuracy: {acc:.2f}%")
        
        comparison_summary[name] = {
            "rmse": rmse,
            "r2_score": r2,
            "accuracy": acc,
            "mae": metrics["regression"]["mae"]
        }
        
        if rmse < lowest_rmse:
            lowest_rmse = rmse
            best_model_name = name
            best_model = model
            best_metrics = metrics
            
    print("\n==================================================")
    print(f" SELECTED BEST MODEL: {best_model_name}")
    print("==================================================")
    
    reg_m = best_metrics["regression"]
    cls_m = best_metrics["classification"]
    
    print("\n--- REGRESSION METRICS ---")
    print(f" - MAE  (Mean Absolute Error):  {reg_m['mae']} days")
    print(f" - MSE  (Mean Squared Error):   {reg_m['mse']}")
    print(f" - RMSE (Root Mean Sq Error):  {reg_m['rmse']} days")
    print(f" - R² Score (Variance Explained): {reg_m['r2_score']}")
    
    print("\n--- STAY CATEGORY CLASSIFICATION METRICS ---")
    print(f" - Stay Category Accuracy: {cls_m['accuracy']}%")
    print(f" - Weighted Precision:     {cls_m['precision']}%")
    print(f" - Weighted Recall:        {cls_m['recall']}%")
    print(f" - Weighted F1 Score:      {cls_m['f1_score']}%")
    print(f" - Confusion Matrix:       {cls_m['confusion_matrix']}")
    
    # 4. Save Model Artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_DIR, "plos_model.pkl")
    preprocessor_save_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    metrics_save_path = os.path.join(MODEL_DIR, "model_metrics.json")
    
    joblib.dump(best_model, model_save_path)
    joblib.dump(preprocessor, preprocessor_save_path)
    
    # Save training metadata and metrics for UI display
    model_metadata = {
        "model_name": best_model_name,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "num_features": int(X_train.shape[1]),
        "regression": reg_m,
        "classification": cls_m,
        "comparison": comparison_summary
    }
    
    with open(metrics_save_path, "w") as f:
        json.dump(model_metadata, f, indent=2)
        
    print(f"\n[+] Trained model saved to '{model_save_path}'.")
    print(f"[+] Preprocessor saved to '{preprocessor_save_path}'.")
    print(f"[+] Performance metrics exported to '{metrics_save_path}'.")
    print("==================================================\n")

if __name__ == "__main__":
    train_and_evaluate()
