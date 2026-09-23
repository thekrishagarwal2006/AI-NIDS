import os
import joblib
import pandas as pd
import numpy as np
from src.shap_explainability import explain_single_prediction

def run_demo():
    print("==========================================")
    print("AI-NIDS DEMONSTRATION & LOCAL EXPLANATION")
    print("==========================================")

    # 1. Load Preprocessor and Best Model
    if not os.path.exists("models/preprocessor.pkl") or not os.path.exists("models/best_model.pkl"):
        print("Error: Models and preprocessor artifacts not found. Please run 'python main.py' first.")
        return

    preprocessor = joblib.load("models/preprocessor.pkl")
    best_model_info = joblib.load("models/best_model.pkl")

    scaler = preprocessor['scaler']
    label_encoder = preprocessor['label_encoder']
    feature_names = preprocessor['feature_names']
    best_model_name = best_model_info['best_model_name']
    model = best_model_info['model']

    print(f"Loaded Preprocessor and Best Model ({best_model_name}).")

    # 2. Load test samples from KDDTest+.txt
    test_path = "data/KDDTest+.txt"
    if not os.path.exists(test_path):
        print("Error: KDDTest+.txt not found.")
        return

    from src.data_preprocessing import COLUMNS, ATTACK_MAPPING, CATEGORICAL_COLS, map_attack_category
    df_test_raw = pd.read_csv(test_path, names=COLUMNS)
    if 'difficulty_level' in df_test_raw.columns:
        df_test_raw = df_test_raw.drop(columns=['difficulty_level'])

    df_test_raw['attack_cat'] = df_test_raw['attack'].apply(map_attack_category)
    X_test_raw = df_test_raw.drop(columns=['attack', 'attack_cat'])
    y_test_raw = df_test_raw['attack_cat']

    # Encode test features matching train pipeline
    X_test_encoded = pd.get_dummies(X_test_raw, columns=CATEGORICAL_COLS)
    
    # Reindex to align with preprocessor train feature names
    train_encoded_cols = preprocessor['train_encoded_columns']
    X_test_aligned = pd.DataFrame(0, index=X_test_encoded.index, columns=train_encoded_cols)
    for col in X_test_encoded.columns:
        if col in train_encoded_cols:
            X_test_aligned[col] = X_test_encoded[col]

    X_test_scaled = scaler.transform(X_test_aligned)

    # Interactive choice or default sample selection
    num_samples = len(X_test_scaled)
    print(f"\nAvailable Test Samples: {num_samples}")
    sample_choice = input(f"Enter sample index to evaluate [0 - {num_samples-1}] (Press Enter for default = 0): ").strip()
    
    if sample_choice.isdigit() and 0 <= int(sample_choice) < num_samples:
        sample_idx = int(sample_choice)
    else:
        sample_idx = 0

    print(f"\nEvaluating Sample Index: {sample_idx}")
    true_label = y_test_raw.iloc[sample_idx]
    print(f"Ground Truth Label: {true_label}")

    sample_vector = X_test_scaled[sample_idx]

    # Predict & Explain
    explanation_str, feat_contribs = explain_single_prediction(
        model, sample_vector, feature_names, label_encoder, sample_idx=sample_idx, results_dir="results/shap"
    )

    print("\n" + explanation_str)

if __name__ == "__main__":
    run_demo()
