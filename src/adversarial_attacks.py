import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Try importing ART components with fallback to custom TensorFlow FGSM/PGD implementations
try:
    from art.estimators.classification import KerasClassifier
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentNumpy
    ART_AVAILABLE = True
except Exception:
    ART_AVAILABLE = False


def generate_fgsm_tf(model, X, y, eps=0.1):
    """
    Custom TensorFlow implementation of Fast Gradient Sign Method (FGSM).
    x_adv = x + eps * sign(grad_x Loss(model(x), y))
    """
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.int64)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        preds = model(X_tensor, training=False)
        loss = loss_fn(y_tensor, preds)

    gradient = tape.gradient(loss, X_tensor)
    signed_grad = tf.sign(gradient)
    X_adv = X_tensor + eps * signed_grad
    return X_adv.numpy()


def generate_pgd_tf(model, X, y, eps=0.1, alpha=0.02, iters=10):
    """
    Custom TensorFlow implementation of Projected Gradient Descent (PGD).
    Iterative FGSM with clipping back into the epsilon-ball around original X.
    """
    X_orig = X.copy()
    X_adv = X.copy()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for _ in range(iters):
        X_tensor = tf.convert_to_tensor(X_adv, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.int64)

        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            preds = model(X_tensor, training=False)
            loss = loss_fn(y_tensor, preds)

        gradient = tape.gradient(loss, X_tensor)
        signed_grad = tf.sign(gradient)

        # Step in direction of gradient
        X_adv = X_adv + alpha * signed_grad.numpy()

        # Project back into epsilon ball
        eta = np.clip(X_adv - X_orig, -eps, eps)
        X_adv = X_orig + eta

    return X_adv


def evaluate_adversarial_robustness(dnn_model, X_test, y_test, epsilons=[0.05, 0.1, 0.2], results_dir="results/adversarial"):
    """
    Evaluate Keras DNN performance on Clean data vs FGSM attack vs PGD attack across different epsilon values.
    """
    os.makedirs(results_dir, exist_ok=True)
    print("\n==========================================")
    print("EVALUATING ADVERSARIAL ROBUSTNESS (FGSM & PGD)")
    print("==========================================")

    # Use a representative sample of test set for fast evaluation (5000 samples)
    eval_size = min(5000, len(X_test))
    np.random.seed(42)
    eval_indices = np.random.choice(len(X_test), eval_size, replace=False)
    X_eval = X_test[eval_indices]
    y_eval = y_test[eval_indices]

    # Clean predictions
    clean_preds = np.argmax(dnn_model.predict(X_eval, verbose=0), axis=1)
    clean_acc = accuracy_score(y_eval, clean_preds)
    print(f"Clean DNN Test Accuracy (sample={eval_size}): {clean_acc * 100:.2f}%")

    adv_results = []
    adv_results.append({
        'Condition': 'Clean',
        'Epsilon': 0.0,
        'Accuracy': clean_acc,
        'Accuracy Drop': 0.0
    })

    # Evaluate across epsilons
    for eps in epsilons:
        print(f"\n--- Testing Epsilon = {eps} ---")
        
        # 1. FGSM Attack
        X_fgsm = generate_fgsm_tf(dnn_model, X_eval, y_eval, eps=eps)
        fgsm_preds = np.argmax(dnn_model.predict(X_fgsm, verbose=0), axis=1)
        fgsm_acc = accuracy_score(y_eval, fgsm_preds)
        fgsm_drop = clean_acc - fgsm_acc
        print(f"FGSM Accuracy (eps={eps}): {fgsm_acc * 100:.2f}% (Drop: {fgsm_drop * 100:.2f}%)")

        adv_results.append({
            'Condition': f'FGSM (eps={eps})',
            'Epsilon': eps,
            'Accuracy': fgsm_acc,
            'Accuracy Drop': fgsm_drop
        })

        # 2. PGD Attack
        X_pgd = generate_pgd_tf(dnn_model, X_eval, y_eval, eps=eps, alpha=eps/5.0, iters=8)
        pgd_preds = np.argmax(dnn_model.predict(X_pgd, verbose=0), axis=1)
        pgd_acc = accuracy_score(y_eval, pgd_preds)
        pgd_drop = clean_acc - pgd_acc
        print(f"PGD Accuracy (eps={eps}):  {pgd_acc * 100:.2f}% (Drop: {pgd_drop * 100:.2f}%)")

        adv_results.append({
            'Condition': f'PGD (eps={eps})',
            'Epsilon': eps,
            'Accuracy': pgd_acc,
            'Accuracy Drop': pgd_drop
        })

    df_adv = pd.DataFrame(adv_results)
    csv_path = os.path.join(results_dir, "adversarial_evaluation.csv")
    df_adv.to_csv(csv_path, index=False)
    print(f"\nSaved adversarial evaluation metrics to {csv_path}")

    # Plot Robustness Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_adv, x='Condition', y='Accuracy', palette='Reds_r')
    plt.title("DNN Accuracy Under Clean vs. Adversarial Attack Conditions", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Attack Condition", fontsize=12)
    plt.xticks(rotation=20)
    plt.ylim(0, 1.05)

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height * 100:.1f}%', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 2), textcoords='offset points')

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "adversarial_performance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved adversarial performance comparison plot to {plot_path}")

    return df_adv
