import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def compute_fpr(cm, normal_index=0):
    """
    Compute False Positive Rate specifically for NIDS:
    FPR = (Normal instances misclassified as Attack) / (Total Normal instances)
    Also returns per-class FPR array.
    """
    # Specific Normal FPR
    total_normal = np.sum(cm[normal_index, :])
    if total_normal == 0:
        normal_fpr = 0.0
    else:
        false_positives = total_normal - cm[normal_index, normal_index]
        normal_fpr = false_positives / total_normal

    # Per-class FPR: FP / (FP + TN)
    num_classes = cm.shape[0]
    per_class_fpr = []
    for i in range(num_classes):
        fp = np.sum(cm[:, i]) - cm[i, i]
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        denom = fp + tn
        fpr = fp / denom if denom > 0 else 0.0
        per_class_fpr.append(fpr)

    return normal_fpr, np.mean(per_class_fpr)

def evaluate_single_model(model, X_test, y_test, model_name, label_encoder, results_dir="results/models"):
    """
    Evaluate a single model, compute metrics, plot confusion matrix.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Predict
    if hasattr(model, "predict_proba"):
        y_pred_probs = model.predict_proba(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
    elif hasattr(model, "predict"):
        preds = model.predict(X_test)
        if preds.ndim > 1 and preds.shape[1] > 1:
            y_pred = np.argmax(preds, axis=1)
        else:
            y_pred = np.round(preds).astype(int).ravel()
    else:
        raise ValueError(f"Model {model_name} has no valid predict method.")

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    
    # Identify index of Normal class
    class_names = list(label_encoder.classes_)
    normal_idx = class_names.index('Normal') if 'Normal' in class_names else 0
    normal_fpr, macro_fpr = compute_fpr(cm, normal_index=normal_idx)

    # Plot & Save Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(results_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"\n--- Evaluation Results for {model_name} ---")
    print(f"Accuracy:        {acc * 100:.2f}%")
    print(f"Macro Precision: {prec_macro * 100:.2f}%")
    print(f"Macro Recall:    {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:  {f1_macro * 100:.2f}%")
    print(f"Normal FPR:      {normal_fpr * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    metrics = {
        'Model': model_name,
        'Accuracy': acc,
        'Macro Precision': prec_macro,
        'Macro Recall': rec_macro,
        'Macro F1-Score': f1_macro,
        'Weighted F1-Score': f1_weighted,
        'Normal FPR': normal_fpr,
        'Macro FPR': macro_fpr
    }
    return metrics, cm

def evaluate_all_models(models_dict, X_test, y_test, label_encoder, results_dir="results/models"):
    """
    Evaluate RF, XGBoost, and DNN on untouched test set.
    Save model_comparison.csv, comparison bar chart, and return best model.
    """
    print("\n==========================================")
    print("EVALUATING ALL MODELS ON UNTOUCHED TEST SET")
    print("==========================================")
    
    results = []
    for model_name, model in models_dict.items():
        metrics, _ = evaluate_single_model(model, X_test, y_test, model_name, label_encoder, results_dir)
        results.append(metrics)

    df_results = pd.DataFrame(results)
    
    # Save model comparison table to results/model_comparison.csv
    csv_path = "results/model_comparison.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved model evaluation comparison to {csv_path}")

    # Plot Model Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    df_melted = pd.melt(
        df_results,
        id_vars=['Model'],
        value_vars=['Accuracy', 'Macro F1-Score', 'Weighted F1-Score'],
        var_name='Metric', value_name='Score'
    )
    ax = sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette='magma')
    plt.title("Model Performance Comparison (Untouched Test Set)", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height * 100:.1f}%', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved model comparison plot to {plot_path}")

    # Select Best Model programmatically based on Weighted F1-Score
    best_row = df_results.sort_values(by='Weighted F1-Score', ascending=False).iloc[0]
    best_model_name = best_row['Model']
    best_model = models_dict[best_model_name]
    
    print(f"\n==========================================")
    print(f"BEST PERFORMING MODEL: {best_model_name} (Weighted F1: {best_row['Weighted F1-Score'] * 100:.2f}%)")
    print(f"==========================================")

    # Persist best model indicator
    os.makedirs('models', exist_ok=True)
    joblib.dump({'best_model_name': best_model_name, 'model': best_model}, 'models/best_model.pkl')

    return best_model_name, best_model, df_results
