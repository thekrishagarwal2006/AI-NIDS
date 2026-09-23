import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# Define standard NSL-KDD column names
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack', 'difficulty_level'
]

# Attack mappings for NSL-KDD (5-class taxonomy)
ATTACK_MAPPING = {
    'normal': 'Normal',
    
    # DoS attacks
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
    'udpstorm': 'DoS', 'worm': 'DoS',
    
    # Probe attacks
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    
    # R2L attacks
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
    'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
    'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    
    # U2R attacks
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'perl': 'U2R',
    'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

def map_attack_category(attack_label):
    clean_label = str(attack_label).strip().rstrip('.')
    return ATTACK_MAPPING.get(clean_label, 'DoS' if 'dos' in clean_label.lower() else 'R2L')

def load_and_preprocess_data(train_path="data/KDDTrain+.txt", test_path="data/KDDTest+.txt", use_smote=True):
    """
    Complete data preprocessing pipeline:
    1. Load NSL-KDD train and test files
    2. Map attacks to 5 categories (Normal, DoS, Probe, R2L, U2R)
    3. Categorical encoding (One-Hot encoding with alignment)
    4. Feature scaling using StandardScaler
    5. Apply SMOTE to training data only
    6. Save preprocessor artifacts to models/
    """
    print("Loading raw NSL-KDD dataset...")
    df_train = pd.read_csv(train_path, names=COLUMNS)
    df_test = pd.read_csv(test_path, names=COLUMNS)

    # Drop difficulty_level if present
    if 'difficulty_level' in df_train.columns:
        df_train = df_train.drop(columns=['difficulty_level'])
    if 'difficulty_level' in df_test.columns:
        df_test = df_test.drop(columns=['difficulty_level'])

    # Map attack labels to 5 categories
    df_train['attack_cat'] = df_train['attack'].apply(map_attack_category)
    df_test['attack_cat'] = df_test['attack'].apply(map_attack_category)

    # Separate X and y
    X_train_raw = df_train.drop(columns=['attack', 'attack_cat'])
    y_train_raw = df_train['attack_cat']
    X_test_raw = df_test.drop(columns=['attack', 'attack_cat'])
    y_test_raw = df_test['attack_cat']

    # One-hot encode categorical features while ensuring train/test column alignment
    X_train_encoded = pd.get_dummies(X_train_raw, columns=CATEGORICAL_COLS)
    X_test_encoded = pd.get_dummies(X_test_raw, columns=CATEGORICAL_COLS)

    # Reindex test set to align columns with train set (fill missing with 0)
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

    feature_names = list(X_train_encoded.columns)

    # Label encode target variable
    label_encoder = LabelEncoder()
    # Explicitly fit on standard 5 classes to ensure order: Normal, DoS, Probe, R2L, U2R
    classes_order = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
    label_encoder.fit(classes_order)

    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    print(f"Preprocessed train shape: {X_train_scaled.shape}, test shape: {X_test_scaled.shape}")

    # Apply SMOTE to training data only
    if use_smote:
        print("Applying SMOTE to training data (test data left untouched)...")
        # To avoid errors with small minority classes like U2R in small samples, set k_neighbors safely
        min_samples = pd.Series(y_train).value_counts().min()
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE training set shape: {X_train_resampled.shape}")
    else:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train

    # Save preprocessing objects
    preprocessor = {
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'categorical_cols': CATEGORICAL_COLS,
        'train_columns': X_train_raw.columns.tolist(),
        'train_encoded_columns': X_train_encoded.columns.tolist()
    }
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("Saved preprocessor pipeline to models/preprocessor.pkl")

    return (
        X_train_resampled, y_train_resampled,
        X_test_scaled, y_test,
        df_train, df_test,
        feature_names, label_encoder
    )
