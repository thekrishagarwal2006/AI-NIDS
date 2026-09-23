import os
import random
import urllib.request
import numpy as np

def set_seed(seed=42):
    """Set random seed for reproducibility across random, numpy, and tensorflow (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def ensure_directories():
    """Ensure all required project directories exist."""
    dirs = [
        "data",
        "models",
        "results",
        "results/eda",
        "results/models",
        "results/shap",
        "results/adversarial",
        "notebooks"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Project directory structure verified.")

def download_nsl_kdd(data_dir="data"):
    """
    Download NSL-KDD dataset (KDDTrain+.txt and KDDTest+.txt) if not present locally.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "KDDTrain+.txt")
    test_path = os.path.join(data_dir, "KDDTest+.txt")

    urls = {
        "KDDTrain+.txt": [
            "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
            "https://raw.githubusercontent.com/merouane-m/NSL-KDD/master/KDDTrain+.txt"
        ],
        "KDDTest+.txt": [
            "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
            "https://raw.githubusercontent.com/merouane-m/NSL-KDD/master/KDDTest+.txt"
        ]
    }

    for file_name, file_urls in urls.items():
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"Downloading {file_name}...")
            download_success = False
            for url in file_urls:
                try:
                    urllib.request.urlretrieve(url, file_path)
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        print(f"Successfully downloaded {file_name} ({os.path.getsize(file_path)} bytes).")
                        download_success = True
                        break
                except Exception as e:
                    print(f"Failed download from {url}: {e}")
            if not download_success:
                raise RuntimeError(
                    f"Could not download {file_name}. Please manually place {file_name} in {data_dir}/"
                )
        else:
            print(f"Dataset file found: {file_path}")

    return train_path, test_path
