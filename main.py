import os
import joblib
import pandas as pd
import numpy as np

from src.utils import set_seed, ensure_directories, download_nsl_kdd
from src.data_preprocessing import load_and_preprocess_data
from src.eda import perform_eda
from src.train_models import train_all_models
from src.evaluate_models import evaluate_all_models
from src.shap_explainability import compute_shap_explanations, explain_single_prediction
from src.adversarial_attacks import evaluate_adversarial_robustness
from src.adversarial_training import perform_adversarial_training

def generate_final_report(best_model_name, df_results, adv_summary, report_path="results/final_report.txt"):
    """
    Generate final summary report with actual empirical results.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    best_row = df_results[df_results['Model'] == best_model_name].iloc[0]
    
    report_content = f"""AI-BASED EXPLAINABLE & ADVERSARIAL-RESISTANT NIDS
=================================================

Dataset:
NSL-KDD (KDDTrain+.txt, KDDTest+.txt)

Models Evaluated:
- Random Forest
- XGBoost
- Keras Deep Neural Network (DNN)

Best Model:
{best_model_name}

Clean Performance (Best Model - {best_model_name}):
- Accuracy:          {best_row['Accuracy'] * 100:.2f}%
- Macro Precision:   {best_row['Macro Precision'] * 100:.2f}%
- Macro Recall:      {best_row['Macro Recall'] * 100:.2f}%
- Macro F1-Score:    {best_row['Macro F1-Score'] * 100:.2f}%
- Weighted F1-Score: {best_row['Weighted F1-Score'] * 100:.2f}%
- False Positive Rate (Normal): {best_row['Normal FPR'] * 100:.2f}%

SHAP Explainability:
Implemented (Global Summary Plot, Global Bar Plot, Local Feature Impact Analysis)

Adversarial Vulnerability (Baseline DNN):
- Clean Accuracy:    {adv_summary['base_clean_acc'] * 100:.2f}%
- FGSM Accuracy:     {adv_summary['base_fgsm_acc'] * 100:.2f}%
- PGD Accuracy:      {adv_summary['base_pgd_acc'] * 100:.2f}%

Adversarial Training & Defense:
- Retrained Clean Accuracy:   {adv_summary['rob_clean_acc'] * 100:.2f}%
- Retrained FGSM Accuracy:    {adv_summary['rob_fgsm_acc'] * 100:.2f}%
- Retrained PGD Accuracy:     {adv_summary['rob_pgd_acc'] * 100:.2f}%
- FGSM Robustness Improvement: +{adv_summary['fgsm_improvement'] * 100:.2f}%
- PGD Robustness Improvement:  +{adv_summary['pgd_improvement'] * 100:.2f}%

Summary Table Across Models:
{df_results.to_string(index=False)}
"""

    with open(report_path, "w") as f:
        f.write(report_content)
        
    print(f"\nFinal report successfully written to {report_path}")
    print(report_content)

def main():
    print("==========================================================================")
    print("AI-BASED EXPLAINABLE & ADVERSARIAL-RESISTANT NIDS (ACADEMIC PROTOTYPE)")
    print("==========================================================================")

    # 1. Set seed and prepare directories
    set_seed(42)
    ensure_directories()

    # 2. Download / Verify NSL-KDD dataset
    train_path, test_path = download_nsl_kdd("data")

    # 3. Load and Preprocess Data
    (
        X_train, y_train,
        X_test, y_test,
        df_train_raw, df_test_raw,
        feature_names, label_encoder
    ) = load_and_preprocess_data(train_path, test_path, use_smote=True)

    # 4. Exploratory Data Analysis
    perform_eda(df_train_raw, results_dir="results/eda")

    # 5. Train Models (Random Forest, XGBoost, DNN)
    models_dict = train_all_models(X_train, y_train, X_test, y_test, models_dir="models")

    # 6. Evaluate Models & Select Best Model
    best_model_name, best_model, df_results = evaluate_all_models(
        models_dict, X_test, y_test, label_encoder, results_dir="results/models"
    )

    # 7. SHAP Explainability
    explainer, shap_matrix, df_sample = compute_shap_explanations(
        best_model, best_model_name, X_test, feature_names, label_encoder, X_train=X_train, results_dir="results/shap"
    )

    # Demonstrate single sample prediction & explanation
    sample_idx = 0
    sample_vector = X_test[sample_idx]
    local_explanation, _ = explain_single_prediction(
        best_model, sample_vector, feature_names, label_encoder, explainer=explainer, sample_idx=sample_idx, results_dir="results/shap"
    )
    print("\nSample Local Explanation Output:")
    print(local_explanation)

    # 8 & 9. Adversarial Robustness Testing (FGSM & PGD on Differentiable DNN)
    dnn_model = models_dict['DNN']
    df_adv_eval = evaluate_adversarial_robustness(
        dnn_model, X_test, y_test, epsilons=[0.05, 0.1, 0.2], results_dir="results/adversarial"
    )

    # 10. Adversarial Training
    robust_dnn, adv_summary = perform_adversarial_training(
        dnn_model, X_train, y_train, X_test, y_test,
        eps=0.1, epochs=15, results_dir="results/adversarial", models_dir="models"
    )

    # 11. Generate Final Report
    generate_final_report(best_model_name, df_results, adv_summary, report_path="results/final_report.txt")

    print("\n==========================================================================")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("All visual plots saved under results/")
    print("Trained models persisted under models/")
    print("==========================================================================")

if __name__ == "__main__":
    main()
