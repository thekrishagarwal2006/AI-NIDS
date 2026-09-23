import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def initialize_explainer(model, model_name, X_background):
    """
    Initialize appropriate SHAP explainer based on model type.
    """
    print(f"Initializing SHAP Explainer for {model_name}...")
    if "Forest" in model_name or "XGBoost" in model_name or hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        # For Neural Networks (DNN), use KernelExplainer with small representative background sample
        bg_sample = shap.sample(X_background, 10)
        
        def predict_fn(x):
            return model.predict(x, verbose=0)

        explainer = shap.KernelExplainer(predict_fn, bg_sample)
    return explainer

def compute_shap_explanations(model, model_name, X_test, feature_names, label_encoder, X_train=None, results_dir="results/shap"):
    """
    Generate global SHAP plots and local sample explanations.
    """
    os.makedirs(results_dir, exist_ok=True)
    print("\n==========================================")
    print(f"GENERATING SHAP EXPLAINABILITY FOR {model_name}")
    print("==========================================")

    num_samples = min(20, len(X_test))
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
    X_sample = X_test[sample_indices]
    df_sample = pd.DataFrame(X_sample, columns=feature_names)

    explainer = initialize_explainer(model, model_name, X_train if X_train is not None else X_test)
    
    # Calculate SHAP values
    if isinstance(explainer, shap.TreeExplainer):
        shap_values = explainer.shap_values(df_sample)
    else:
        shap_values = explainer.shap_values(df_sample, nsamples=25)

    # Standardize shap_values format across multi-class models
    if isinstance(shap_values, list):
        shap_vals_matrix = shap_values
    elif hasattr(shap_values, "values"):
        shap_vals_matrix = shap_values.values
    else:
        shap_vals_matrix = shap_values

    # 1. Global SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    if isinstance(shap_vals_matrix, list):
        shap.summary_plot(shap_vals_matrix, df_sample, class_names=label_encoder.classes_, show=False)
    elif shap_vals_matrix.ndim == 3:
        shap.summary_plot(shap_vals_matrix[:, :, 0], df_sample, show=False)
    else:
        shap.summary_plot(shap_vals_matrix, df_sample, show=False)

    plt.title(f"Global SHAP Feature Importance Summary ({model_name})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    summary_path = os.path.join(results_dir, "shap_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved global SHAP summary plot to {summary_path}")

    # 2. Global SHAP Bar Plot
    plt.figure(figsize=(10, 6))
    if isinstance(shap_vals_matrix, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals_matrix], axis=0)
    elif shap_vals_matrix.ndim == 3:
        mean_abs_shap = np.abs(shap_vals_matrix).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(shap_vals_matrix).mean(axis=0)

    top_indices = np.argsort(mean_abs_shap)[::-1][:15]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = mean_abs_shap[top_indices]

    plt.barh(range(len(top_features)), top_scores[::-1], align='center', color='skyblue')
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=12)
    plt.title(f"Top 15 Most Important Features - Global SHAP ({model_name})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    bar_path = os.path.join(results_dir, "shap_bar.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved global SHAP bar plot to {bar_path}")

    return explainer, shap_vals_matrix, df_sample

def explain_single_prediction(model, sample_vector, feature_names, label_encoder, explainer=None, sample_idx=0, results_dir="results/shap"):
    """
    Generate local human-readable explanation and visual feature contribution plot for a single instance.
    """
    os.makedirs(results_dir, exist_ok=True)
    sample_df = pd.DataFrame([sample_vector], columns=feature_names)

    # Get model prediction and class probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample_df)[0]
    elif hasattr(model, "predict"):
        p = model.predict(sample_df, verbose=0)
        probs = p[0] if p.ndim > 1 else np.array([1 - p[0], p[0]])
    else:
        probs = np.ones(len(label_encoder.classes_)) / len(label_encoder.classes_)

    pred_class_idx = np.argmax(probs)
    pred_class_name = label_encoder.inverse_transform([pred_class_idx])[0]
    confidence = probs[pred_class_idx]

    # Calculate local SHAP values
    if explainer is None:
        explainer = initialize_explainer(model, "Best Model", np.array([sample_vector]))

    if isinstance(explainer, shap.TreeExplainer):
        sv_raw = explainer.shap_values(sample_df)
        if isinstance(sv_raw, list):
            sv = sv_raw[pred_class_idx][0]
        elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
            sv = sv_raw[0, :, pred_class_idx]
        else:
            sv = sv_raw[0]
    else:
        sv_raw = explainer.shap_values(sample_df, nsamples=25)
        if isinstance(sv_raw, list):
            sv = sv_raw[pred_class_idx][0]
        elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
            sv = sv_raw[0, :, pred_class_idx]
        elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 2:
            sv = sv_raw[0]
        else:
            sv = sv_raw

    # Ensure sv is a 1D array of float scalars
    sv = np.asarray(sv, dtype=float).ravel()

    # Rank features by absolute SHAP contribution
    feature_contributions = [(feature_names[i], float(sample_vector[i]), float(sv[i])) for i in range(len(feature_names))]
    feature_contributions.sort(key=lambda x: abs(x[2]), reverse=True)

    # Human-readable explanation text
    explanation_text = []
    explanation_text.append("========================================")
    explanation_text.append("NIDS PREDICTION EXPLANATION (LOCAL SHAP)")
    explanation_text.append("========================================")
    explanation_text.append(f"Prediction:  {pred_class_name.upper()}")
    explanation_text.append(f"Confidence:  {confidence * 100:.2f}%")
    explanation_text.append("\nTop contributing features:")

    for idx, (feat_name, val, shap_val) in enumerate(feature_contributions[:5], 1):
        direction = "increased" if shap_val > 0 else "decreased"
        sign = "+" if shap_val > 0 else ""
        explanation_text.append(
            f"{idx}. {feat_name} : {sign}{shap_val:.4f} ({direction} probability of {pred_class_name})"
        )

    explanation_text.append("\nExplanation:")
    top_pos = [f[0] for f in feature_contributions if f[2] > 0][:2]
    if top_pos:
        explanation_text.append(
            f"The prediction was primarily influenced by\nthe above network-flow features ({', '.join(top_pos)})."
        )
    else:
        explanation_text.append("The prediction was primarily influenced by\nthe above network-flow features.")

    full_explanation = "\n".join(explanation_text)

    # Local visualization plot
    plt.figure(figsize=(9, 5))
    top_10 = feature_contributions[:10]
    names = [x[0] for x in top_10]
    vals = [x[2] for x in top_10]
    colors = ['crimson' if v > 0 else 'navy' for v in vals]

    plt.barh(range(len(names)), vals[::-1], color=colors[::-1], align='center')
    plt.yticks(range(len(names)), names[::-1])
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel(f"SHAP Value (Contribution to '{pred_class_name}' Prediction)", fontsize=11)
    plt.title(f"Local Feature Impact for Sample Prediction: {pred_class_name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    local_plot_path = os.path.join(results_dir, f"local_explanation_sample_{sample_idx}.png")
    plt.savefig(local_plot_path, dpi=300)
    plt.close()

    return full_explanation, feature_contributions
