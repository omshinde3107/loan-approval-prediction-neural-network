from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import threading
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Artifact paths (filenames are the ones train_model.py writes)
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")

# --------------------------------------------------------------------------- #
# Column configuration — copied from train_model.py (identical in app.py and
# test_model.py). Do not change without retraining.
# --------------------------------------------------------------------------- #
TARGET_COLUMN = "Loan_Status"

CATEGORICAL_COLUMNS: List[str] = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

NUMERICAL_COLUMNS: List[str] = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

# Exact feature order used for training (Loan_ID and Loan_Status excluded).
FEATURE_COLUMNS: List[str] = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

# Decision threshold used by train_model.py when evaluating (proba >= 0.5).
DECISION_THRESHOLD = 0.5

# The label the target encoder maps to 1 in training ("Y" in the dataset).
# It is verified against the saved encoder at load time rather than assumed.
APPROVED_LABEL = "Y"


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class PredictionInputError(ValueError):
    """The applicant data is missing, malformed, or has an unseen category."""


class ArtifactLoadError(RuntimeError):
    """The model / scaler / encoders could not be loaded or are inconsistent."""


# --------------------------------------------------------------------------- #
# Lazy, thread-safe artifact loading (importing this module stays cheap and
# never raises just because artifacts are absent)
# --------------------------------------------------------------------------- #
_lock = threading.Lock()
_artifacts: Optional[Dict[str, Any]] = None


def _load_pickle(path: str, description: str) -> Any:
    if not os.path.exists(path):
        raise ArtifactLoadError(
            f"{description} not found at '{path}'. Run 'python train_model.py' first."
        )
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        raise ArtifactLoadError(f"Failed to load {description} from '{path}': {exc}") from exc


def load_artifacts(force_reload: bool = False) -> Dict[str, Any]:
    """
    Load and cache the model, scaler and label encoders.

    Returns a dict with keys: "model", "scaler", "encoders".
    Raises ArtifactLoadError if anything is missing or inconsistent.
    """
    global _artifacts
    with _lock:
        if _artifacts is not None and not force_reload:
            return _artifacts

        # Model ----------------------------------------------------------------
        if not os.path.exists(MODEL_PATH):
            raise ArtifactLoadError(
                f"Model file not found at '{MODEL_PATH}'. Run 'python train_model.py' first."
            )
        try:
            from tensorflow import keras  # imported lazily on purpose

            model = keras.models.load_model(MODEL_PATH)
        except Exception as exc:
            raise ArtifactLoadError(f"Failed to load model from '{MODEL_PATH}': {exc}") from exc

        # Scaler + encoders ----------------------------------------------------
        scaler = _load_pickle(SCALER_PATH, "scaler")
        encoders = _load_pickle(ENCODERS_PATH, "label encoders")

        # Consistency checks ---------------------------------------------------
        if not (hasattr(scaler, "mean_") and hasattr(scaler, "scale_")):
            raise ArtifactLoadError("Loaded scaler is not fitted (missing mean_/scale_).")
        if scaler.mean_.shape[0] != len(NUMERICAL_COLUMNS):
            raise ArtifactLoadError(
                f"Scaler was fit on {scaler.mean_.shape[0]} numeric feature(s) but "
                f"{len(NUMERICAL_COLUMNS)} are expected: {NUMERICAL_COLUMNS}."
            )
        scaler_names = getattr(scaler, "feature_names_in_", None)
        if scaler_names is not None and list(scaler_names) != NUMERICAL_COLUMNS:
            raise ArtifactLoadError(
                f"Scaler feature names {list(scaler_names)} do not match "
                f"NUMERICAL_COLUMNS {NUMERICAL_COLUMNS}."
            )

        if not isinstance(encoders, dict):
            raise ArtifactLoadError("label_encoders.pkl did not contain a dict of encoders.")
        missing = [c for c in CATEGORICAL_COLUMNS + [TARGET_COLUMN] if c not in encoders]
        if missing:
            raise ArtifactLoadError(f"label_encoders.pkl is missing encoder(s) for: {missing}.")

        model_input_dim = model.input_shape[-1]
        if model_input_dim != len(FEATURE_COLUMNS):
            raise ArtifactLoadError(
                f"Model expects {model_input_dim} input feature(s) but "
                f"{len(FEATURE_COLUMNS)} are defined in FEATURE_COLUMNS."
            )

        # Target interpretation: the sigmoid output is P(class encoded as 1).
        # Confirm that class is the approval label ("Y") instead of assuming.
        target_classes = [str(c) for c in encoders[TARGET_COLUMN].classes_]
        if APPROVED_LABEL not in target_classes:
            raise ArtifactLoadError(
                f"Target encoder classes {target_classes} do not contain '{APPROVED_LABEL}'."
            )
        approved_code = int(encoders[TARGET_COLUMN].transform([APPROVED_LABEL])[0])
        if approved_code != 1:
            raise ArtifactLoadError(
                f"Target encoder maps '{APPROVED_LABEL}' to {approved_code}, but the model's "
                f"sigmoid output represents class 1. Retrain or review the encoder."
            )

        _artifacts = {"model": model, "scaler": scaler, "encoders": encoders}
        return _artifacts


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _validate_and_coerce(applicant: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Validate a raw applicant mapping and return cleaned values:
    categorical -> stripped str, numeric -> float.
    Extra keys (e.g. Loan_ID) are ignored.
    """
    if not isinstance(applicant, Mapping):
        raise PredictionInputError(
            f"Applicant data must be a mapping/dict of field names to values "
            f"(got {type(applicant).__name__})."
        )

    missing_fields = []
    clean: Dict[str, Any] = {}

    for col in CATEGORICAL_COLUMNS:
        value = applicant.get(col)
        value = "" if value is None else str(value).strip()
        if value == "":
            missing_fields.append(col)
        else:
            clean[col] = value

    for col in NUMERICAL_COLUMNS:
        value = applicant.get(col)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            missing_fields.append(col)
            continue
        if isinstance(value, bool):
            raise PredictionInputError(f"'{col}' must be a number, not a boolean.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise PredictionInputError(f"'{col}' must be a valid number (got '{value}').")
        if not math.isfinite(number):
            raise PredictionInputError(f"'{col}' must be a finite number (got '{value}').")
        clean[col] = number

    if missing_fields:
        raise PredictionInputError(
            "Missing required field(s): " + ", ".join(missing_fields) + "."
        )

    # Range checks (same rules as app.py)
    if clean["ApplicantIncome"] < 0 or clean["CoapplicantIncome"] < 0:
        raise PredictionInputError("Income values cannot be negative.")
    if clean["LoanAmount"] <= 0:
        raise PredictionInputError("Loan amount must be greater than 0.")
    if clean["Loan_Amount_Term"] <= 0:
        raise PredictionInputError("Loan term must be greater than 0.")
    if clean["Credit_History"] not in (0.0, 1.0):
        raise PredictionInputError("Credit history must be either 0 (poor) or 1 (good).")

    return clean


# --------------------------------------------------------------------------- #
# Preprocessing (mirrors train_model.py)
# --------------------------------------------------------------------------- #
def preprocess(applicant: Mapping[str, Any], artifacts: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Validate + preprocess one applicant into a float32 array of shape
    (1, len(FEATURE_COLUMNS)), ready for model inference.
    """
    artifacts = artifacts or load_artifacts()
    encoders = artifacts["encoders"]
    scaler = artifacts["scaler"]

    clean = _validate_and_coerce(applicant)

    # 1. Label-encode categoricals with the saved encoders
    encoded: Dict[str, Any] = dict(clean)
    for col in CATEGORICAL_COLUMNS:
        encoder = encoders[col]
        try:
            encoded[col] = int(encoder.transform([clean[col]])[0])
        except ValueError:
            valid = ", ".join(str(c) for c in encoder.classes_)
            raise PredictionInputError(
                f"Unrecognized value '{clean[col]}' for '{col}'. Valid options: {valid}."
            )

    # 2. Assemble the row in the exact training feature order
    row = pd.DataFrame([[encoded[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    # 3. Scale ONLY the numerical columns (DataFrame keeps the feature names the
    #    scaler was fit with, avoiding scikit-learn feature-name warnings)
    row[NUMERICAL_COLUMNS] = scaler.transform(row[NUMERICAL_COLUMNS])

    return row[FEATURE_COLUMNS].astype("float32").values


# --------------------------------------------------------------------------- #
# Public prediction API
# --------------------------------------------------------------------------- #
def predict_loan(applicant: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Predict loan approval for a single applicant.

    Args:
        applicant: mapping containing every field in FEATURE_COLUMNS.

    Returns:
        {
            "prediction":  "Loan Approved" | "Loan Rejected",
            "approved":    bool,
            "probability": float,   # model output = P(approved), 0..1
            "confidence":  float,   # % confidence in the predicted class
        }

    Raises:
        PredictionInputError: invalid / missing / unrecognized input.
        ArtifactLoadError:    model artifacts missing or inconsistent.
    """
    artifacts = load_artifacts()
    features = preprocess(applicant, artifacts)

    output = artifacts["model"](features, training=False)
    probability = float(np.asarray(output).reshape(-1)[0])
    if not math.isfinite(probability):
        raise RuntimeError("Model returned a non-finite prediction.")

    approved = probability >= DECISION_THRESHOLD
    confidence = probability if approved else 1.0 - probability

    return {
        "prediction": "Loan Approved" if approved else "Loan Rejected",
        "approved": bool(approved),
        "probability": round(probability, 4),
        "confidence": round(confidence * 100, 2),
    }


def get_schema() -> Dict[str, Any]:
    """Describe the expected input, using the actual saved encoders for valid options."""
    encoders = load_artifacts()["encoders"]
    return {
        "feature_order": FEATURE_COLUMNS,
        "categorical": {c: [str(v) for v in encoders[c].classes_] for c in CATEGORICAL_COLUMNS},
        "numeric": NUMERICAL_COLUMNS,
        "notes": {"Credit_History": "0 (poor) or 1 (good)", "all_fields": "required"},
    }


# --------------------------------------------------------------------------- #
# Command-line interface (applicant data comes from YOU, none is built in)
# --------------------------------------------------------------------------- #
def _main() -> int:
    parser = argparse.ArgumentParser(description="Predict loan approval for one applicant.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", help="Applicant as a JSON object string.")
    group.add_argument("--file", help="Path to a JSON file containing one applicant object.")
    group.add_argument("--schema", action="store_true", help="Print expected fields and valid categories.")
    args = parser.parse_args()

    try:
        if args.schema:
            print(json.dumps(get_schema(), indent=2))
            return 0

        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                applicant = json.load(f)
        else:
            applicant = json.loads(args.json)

        print(json.dumps(predict_loan(applicant), indent=2))
        return 0

    except (json.JSONDecodeError, OSError) as exc:
        print(f"Input error: could not read applicant JSON ({exc}).")
        return 2
    except PredictionInputError as exc:
        print(f"Input error: {exc}")
        return 2
    except ArtifactLoadError as exc:
        print(f"Artifact error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())