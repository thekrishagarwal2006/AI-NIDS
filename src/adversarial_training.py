import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from src.train_models import build_dnn_architecture
from src.adversarial_attacks import generate_fgsm_tf, generate_pgd_tf

def perform_adversarial_training(
    baseline_dnn, X_train, y_train, X_test, y_test,
    eps=0.1, epochs=8, batch_size=256, random_state=42,
    results_dir="results/adversarial", models_dir="models"
):
    """
    Pipeline for Adversarial Training:
    1. Generate FGSM and PGD adversarial samples from training data
    2. Concatenate clean + adversarial training sets
    3. Retrain DNN model on augmented dataset
    4. Evaluate retrained model on clean, FGSM, and PGD test sets
    5. Compute Robust Accuracy Improvement
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("\n==========================================")
    print("PERFORMING ADVERSARIAL TRAINING")
    print("==========================================")

    # Use representative test sample (5000 samples)
    eval_size = min(5000, len(X_test))
    np.random.seed(42)
    eval_indices = np.random.choice(len(X_test), eval_size, replace=False)
    X_test_eval = X_test[eval_indices]
    y_test_eval = y_test[eval_indices]

    # Evaluate Baseline DNN performance first
    clean_preds_base = np.argmax(baseline_dnn.predict(X_test_eval, verbose=0), axis=1)
    base_clean_acc = accuracy_score(y_test_eval, clean_preds_base)

    X_test_fgsm = generate_fgsm_tf(baseline_dnn, X_test_eval, y_test_eval, eps=eps)
    fgsm_preds_base = np.argmax(baseline_dnn.predict(X_test_fgsm, verbose=0), axis=1)
    base_fgsm_acc = accuracy_score(y_test_eval, fgsm_preds_base)

    X_test_pgd = generate_pgd_tf(baseline_dnn, X_test_eval, y_test_eval, eps=eps, alpha=eps/5.0, iters=8)
    pgd_preds_base = np.argmax(baseline_dnn.predict(X_test_pgd, verbose=0), axis=1)
    base_pgd_acc = accuracy_score(y_test_eval, pgd_preds_base)

    print(f"Baseline DNN - Clean Acc: {base_clean_acc * 100:.2f}%, FGSM Acc: {base_fgsm_acc * 100:.2f}%, PGD Acc: {base_pgd_acc * 100:.2f}%")

    # Step 1: Generate adversarial samples from training data
    print(f"\nGenerating FGSM & PGD training samples (eps={eps})...")
    sample_size = min(15000, len(X_train))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sub = X_train[indices]
    y_train_sub = y_train[indices]

    X_train_fgsm = generate_fgsm_tf(baseline_dnn, X_train_sub, y_train_sub, eps=eps)
    X_train_pgd = generate_pgd_tf(baseline_dnn, X_train_sub, y_train_sub, eps=eps, alpha=eps/5.0, iters=5)

    # Step 2: Combine clean + adversarial samples
    X_train_augmented = np.vstack([X_train, X_train_fgsm, X_train_pgd])
    y_train_augmented = np.hstack([y_train, y_train_sub, y_train_sub])

    print(f"Augmented training set shape: {X_train_augmented.shape} (Original: {X_train.shape})")

    # Step 3: Retrain DNN model
    print("\nRetraining DNN model on augmented dataset...")
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    robust_dnn = build_dnn_architecture(input_dim, num_classes)

    robust_dnn.fit(
        X_train_augmented, y_train_augmented,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_eval, y_test_eval),
        verbose=1
    )

    # Step 4: Evaluate Retrained Robust DNN
    clean_preds_rob = np.argmax(robust_dnn.predict(X_test_eval, verbose=0), axis=1)
    rob_clean_acc = accuracy_score(y_test_eval, clean_preds_rob)

    X_test_fgsm_rob = generate_fgsm_tf(robust_dnn, X_test_eval, y_test_eval, eps=eps)
    fgsm_preds_rob = np.argmax(robust_dnn.predict(X_test_fgsm_rob, verbose=0), axis=1)
    rob_fgsm_acc = accuracy_score(y_test_eval, fgsm_preds_rob)

    X_test_pgd_rob = generate_pgd_tf(robust_dnn, X_test_eval, y_test_eval, eps=eps, alpha=eps/5.0, iters=8)
    pgd_preds_rob = np.argmax(robust_dnn.predict(X_test_pgd_rob, verbose=0), axis=1)
    rob_pgd_acc = accuracy_score(y_test_eval, pgd_preds_rob)

    # Step 5: Compute Robust Accuracy Improvement
    fgsm_improvement = rob_fgsm_acc - base_fgsm_acc
    pgd_improvement = rob_pgd_acc - base_pgd_acc

    print("\n==========================================")
    print("ADVERSARIAL TRAINING EVALUATION SUMMARY")
    print("==========================================")
    print(f"Clean Accuracy:               Baseline = {base_clean_acc * 100:.2f}% | Robust = {rob_clean_acc * 100:.2f}%")
    print(f"FGSM Accuracy (eps={eps}):      Baseline = {base_fgsm_acc * 100:.2f}% | Robust = {rob_fgsm_acc * 100:.2f}% (Gain: +{fgsm_improvement * 100:.2f}%)")
    print(f"PGD Accuracy (eps={eps}):       Baseline = {base_pgd_acc * 100:.2f}% | Robust = {rob_pgd_acc * 100:.2f}% (Gain: +{pgd_improvement * 100:.2f}%)")

    # Save robust model
    robust_model_path = os.path.join(models_dir, "adversarial_dnn.keras")
    robust_dnn.save(robust_model_path)
    print(f"Saved adversarially trained DNN to {robust_model_path}")

    # Plot Comparison
    comparison_data = pd.DataFrame([
        {'Model': 'Baseline DNN', 'Condition': 'Clean', 'Accuracy': base_clean_acc},
        {'Model': 'Baseline DNN', 'Condition': f'FGSM (eps={eps})', 'Accuracy': base_fgsm_acc},
        {'Model': 'Baseline DNN', 'Condition': f'PGD (eps={eps})', 'Accuracy': base_pgd_acc},
        {'Model': 'Adversarially Trained DNN', 'Condition': 'Clean', 'Accuracy': rob_clean_acc},
        {'Model': 'Adversarially Trained DNN', 'Condition': f'FGSM (eps={eps})', 'Accuracy': rob_fgsm_acc},
        {'Model': 'Adversarially Trained DNN', 'Condition': f'PGD (eps={eps})', 'Accuracy': rob_pgd_acc},
    ])

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=comparison_data, x='Condition', y='Accuracy', hue='Model', palette='Set1')
    plt.title("Robustness Comparison: Baseline DNN vs. Adversarially Trained DNN", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Evaluation Condition", fontsize=12)
    plt.ylim(0, 1.05)

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height * 100:.1f}%', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')

    plt.tight_layout()
    comp_plot_path = os.path.join(results_dir, "adversarial_training_comparison.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"Saved adversarial training comparison plot to {comp_plot_path}")

    summary = {
        'base_clean_acc': base_clean_acc,
        'base_fgsm_acc': base_fgsm_acc,
        'base_pgd_acc': base_pgd_acc,
        'rob_clean_acc': rob_clean_acc,
        'rob_fgsm_acc': rob_fgsm_acc,
        'rob_pgd_acc': rob_pgd_acc,
        'fgsm_improvement': fgsm_improvement,
        'pgd_improvement': pgd_improvement
    }
    return robust_dnn, summary
