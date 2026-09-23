import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def train_random_forest(X_train, y_train, random_state=42):
    """
    Train Model 1: Random Forest Baseline.
    """
    print("\n--- Training Model 1: Random Forest ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)
    print("Random Forest training complete.")
    return rf_model

def train_xgboost(X_train, y_train, random_state=42):
    """
    Train Model 2: XGBoost Classifier.
    """
    print("\n--- Training Model 2: XGBoost ---")
    num_classes = len(np.unique(y_train))
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        eval_metric='mlogloss',
        random_state=random_state,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    print("XGBoost training complete.")
    return xgb_model

def build_dnn_architecture(input_dim, num_classes=5):
    """
    Build simple Deep Neural Network:
    Input -> Dense(128, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.3) -> Dense(5, softmax)
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_dnn(X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=256, random_state=42):
    """
    Train Model 3: Keras Deep Neural Network (DNN).
    """
    print("\n--- Training Model 3: Keras Deep Neural Network (DNN) ---")
    tf.random.set_seed(random_state)
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    dnn_model = build_dnn_architecture(input_dim, num_classes)
    
    callbacks = [
        EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', patience=5, restore_best_weights=True)
    ]
    
    validation_data = (X_val, y_val) if (X_val is not None and y_val is not None) else None
    
    dnn_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )
    print("Keras DNN training complete.")
    return dnn_model

def train_all_models(X_train, y_train, X_test, y_test, models_dir="models"):
    """
    Train all 3 models and persist them in models/ directory.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Random Forest
    rf_model = train_random_forest(X_train, y_train)
    joblib.dump(rf_model, os.path.join(models_dir, "random_forest.pkl"))
    
    # 2. XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    joblib.dump(xgb_model, os.path.join(models_dir, "xgboost.pkl"))
    
    # 3. Keras DNN
    dnn_model = train_dnn(X_train, y_train, X_val=X_test, y_val=y_test, epochs=20)
    dnn_model.save(os.path.join(models_dir, "dnn_model.keras"))
    
    print("\nAll models trained and saved to models/ directory.")
    return {
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "DNN": dnn_model
    }
