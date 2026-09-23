import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def perform_eda(df_train, results_dir="results/eda"):
    """
    Perform Exploratory Data Analysis and save plots into results/eda/.
    """
    os.makedirs(results_dir, exist_ok=True)
    print("Performing Exploratory Data Analysis (EDA)...")

    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 10})

    # 1. Class Distribution (5 Attack Categories)
    plt.figure(figsize=(8, 5))
    cat_counts = df_train['attack_cat'].value_counts()
    ax = sns.barplot(x=cat_counts.index, y=cat_counts.values, palette="viridis")
    plt.title("NSL-KDD Attack Category Distribution (Training Data)", fontsize=14, fontweight='bold')
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Sample Count", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plot_path1 = os.path.join(results_dir, "attack_category_distribution.png")
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"Saved {plot_path1}")

    # 2. Top 10 Specific Attack Types Distribution
    plt.figure(figsize=(10, 6))
    top_attacks = df_train['attack'].value_counts().head(10)
    ax = sns.barplot(x=top_attacks.values, y=top_attacks.index, palette="mako")
    plt.title("Top 10 Specific Traffic Types in NSL-KDD Train Set", fontsize=14, fontweight='bold')
    plt.xlabel("Sample Count", fontsize=12)
    plt.ylabel("Traffic Type", fontsize=12)
    plt.tight_layout()
    plot_path2 = os.path.join(results_dir, "top_attack_types.png")
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"Saved {plot_path2}")

    # 3. Correlation Analysis for Key Numerical Features
    num_cols = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
                'serror_rate', 'same_srv_rate', 'diff_srv_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate']
    available_num_cols = [c for c in num_cols if c in df_train.columns]
    
    plt.figure(figsize=(10, 8))
    corr = df_train[available_num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
    plt.title("Feature Correlation Matrix (Key Numerical Features)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path3 = os.path.join(results_dir, "feature_correlations.png")
    plt.savefig(plot_path3, dpi=300)
    plt.close()
    print(f"Saved {plot_path3}")

    # 4. Feature Distributions by Attack Category
    plt.figure(figsize=(12, 6))
    if 'src_bytes' in df_train.columns and 'count' in df_train.columns:
        # Log scale for bytes/count to handle extreme skewness
        df_plot = df_train.copy()
        df_plot['log_src_bytes'] = np.log1p(df_plot['src_bytes'])
        df_plot['log_count'] = np.log1p(df_plot['count'])
        
        sns.scatterplot(
            data=df_plot, x='log_src_bytes', y='log_count',
            hue='attack_cat', alpha=0.7, palette="Set2", s=20
        )
        plt.title("Log Source Bytes vs. Connection Count by Traffic Class", fontsize=14, fontweight='bold')
        plt.xlabel("Log(Source Bytes + 1)", fontsize=12)
        plt.ylabel("Log(Connection Count + 1)", fontsize=12)
        plt.legend(title="Class")
        plt.tight_layout()
        plot_path4 = os.path.join(results_dir, "feature_scatter_distribution.png")
        plt.savefig(plot_path4, dpi=300)
        plt.close()
        print(f"Saved {plot_path4}")

    print("EDA completed successfully.")
