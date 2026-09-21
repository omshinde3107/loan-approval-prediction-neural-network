"""
train_model.py
================
Loan Approval Prediction Using Neural Networks

This script:
  1. Loads loan_approval_dataset.csv
  2. Cleans/preprocesses the data (missing values, encoding, scaling)
  3. Builds and trains a feed-forward neural network (Keras)
  4. Evaluates the model (Accuracy, Precision, Recall, F1, Confusion Matrix)
  5. Persists the trained model + preprocessing objects to disk:
       - loan_model.keras
       - scaler.pkl
       - label_encoders.pkl

Run:
    python train_model.py
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "loan_approval_dataset.csv"
MODEL_PATH = "loan_model.keras"
SCALER_PATH = "scaler.pkl"
ENCODERS_PATH = "label_encoders.pkl"

TEST_SIZE = 0.2
EPOCHS = 50
BATCH_SIZE = 32

# Columns present in the CSV but not useful as model features
ID_COLUMNS = ["Loan_ID"]

# Target column
TARGET_COLUMN = "Loan_Status"

# Categorical columns that need label encoding
CATEGORICAL_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

# Numerical columns that need scaling
NUMERICAL_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    df = pd.read_csv(path)

    # Drop identifier columns that carry no predictive signal
    for col in ID_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


# ---------------------------------------------------------------------------
# 2. Handle missing values
# ---------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in a sensible, column-appropriate way:
      - Categorical columns  -> mode (most frequent category)
      - Numerical columns    -> median (robust to outliers/skew)
    """
    df = df.copy()

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            mode_value = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_value)

    for col in NUMERICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    return df


# ---------------------------------------------------------------------------
# 3. Encode categorical features
# ---------------------------------------------------------------------------
def encode_categorical_features(df: pd.DataFrame):
    """
    Label-encode every categorical column (including the target).
    Returns the transformed DataFrame plus a dict of fitted encoders
    so the same mapping can be reused at inference time.
    """
    df = df.copy()
    encoders = {}

    # Encode feature columns
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Encode the target column separately (Y/N -> 1/0)
    target_le = LabelEncoder()
    df[TARGET_COLUMN] = target_le.fit_transform(df[TARGET_COLUMN].astype(str))
    encoders[TARGET_COLUMN] = target_le

    return df, encoders


# ---------------------------------------------------------------------------
# 4. Build the neural network
# ---------------------------------------------------------------------------
def build_model(input_dim: int) -> keras.Model:
    """
    Feed-forward neural network:
        Input -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Sigmoid)
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,), name="input_layer"),
            layers.Dense(32, activation="relu", name="hidden_layer_1"),
            layers.Dense(16, activation="relu", name="hidden_layer_2"),
            layers.Dense(1, activation="sigmoid", name="output_layer"),
        ],
        name="loan_approval_nn",
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------
def main():
    # --- Load ---------------------------------------------------------
    print("Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    # --- Clean ---------------------------------------------------------
    print("Handling missing values...")
    df = handle_missing_values(df)
    print(f"Remaining missing values: {int(df.isnull().sum().sum())}")

    # --- Encode ---------------------------------------------------------
    print("Encoding categorical features...")
    df, label_encoders = encode_categorical_features(df)

    # --- Split features / target ---------------------------------------
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values

    # Keep a consistent column order for downstream inference
    feature_columns = X.columns.tolist()

    # --- Train / test split ---------------------------------------------
    print(f"Splitting data ({int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # --- Scale numerical features ----------------------------------------
    # Fit the scaler ONLY on training data to avoid data leakage,
    # then apply the same transformation to the test set.
    print("Scaling numerical features...")
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[NUMERICAL_COLUMNS] = scaler.fit_transform(X_train[NUMERICAL_COLUMNS])
    X_test_scaled[NUMERICAL_COLUMNS] = scaler.transform(X_test[NUMERICAL_COLUMNS])

    # Ensure everything is float32 for TensorFlow
    X_train_scaled = X_train_scaled[feature_columns].astype("float32").values
    X_test_scaled = X_test_scaled[feature_columns].astype("float32").values

    # --- Build model ---------------------------------------------------
    print("Building neural network...")
    model = build_model(input_dim=X_train_scaled.shape[1])
    model.summary()

    # --- Train -----------------------------------------------------------
    print(f"\nTraining for {EPOCHS} epochs...")
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )

    # --- Evaluate ----------------------------------------------------------
    print("\nEvaluating on test set...")
    y_pred_proba = model.predict(X_test_scaled).ravel()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print("                 Predicted N   Predicted Y")
    print(f"Actual N         {cm[0][0]:<13}{cm[0][1]}")
    print(f"Actual Y         {cm[1][0]:<13}{cm[1][1]}")
    print("\nDetailed classification report:")
    target_names = label_encoders[TARGET_COLUMN].classes_.astype(str)
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    print("=" * 50)

    # --- Persist model + preprocessing objects -----------------------------
    print("\nSaving artifacts...")

    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {SCALER_PATH}")

    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(label_encoders, f)
    print(f"Label encoders saved to: {ENCODERS_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
