"""
app.py
======
Flask backend for "Loan Approval Prediction Using Neural Networks".

This app:
  1. Loads the trained Keras model (loan_model.keras), the fitted
     StandardScaler (scaler.pkl), and the fitted LabelEncoders
     (label_encoders.pkl) produced by train_model.py.
  2. Serves the application form on GET "/" (templates/index.html).
  3. Accepts a JSON POST on "/predict" — index.html's form is submitted
     via JavaScript `fetch()` with a JSON body, NOT a standard HTML form
     POST, so this endpoint reads `request.get_json()` (with a fallback
     to form-encoded data for compatibility with tools like curl/Postman
     or a plain HTML form).
  4. Pre-processes the input EXACTLY the way train_model.py pre-processed
     the training data, runs the neural network, and returns the result
     as JSON: a decision label, an "approved" boolean, and a confidence
     percentage.

Column definitions below are copied verbatim from train_model.py's
CATEGORICAL_COLUMNS / NUMERICAL_COLUMNS / FEATURE_COLUMNS so the feature
set and order used here can never silently drift from what the model was
actually trained on.

Run:
    python app.py
Then visit http://127.0.0.1:5000/
"""

import os
import pickle
import traceback

import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow import keras

# --------------------------------------------------------------------------- #
# Paths to the artifacts produced by train_model.py
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")

# --------------------------------------------------------------------------- #
# Column configuration — MUST mirror train_model.py exactly, since the
# scaler / encoders / model were all fit using these definitions.
# --------------------------------------------------------------------------- #
CATEGORICAL_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

NUMERICAL_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

# The exact feature order train_model.py trained on (features only, after
# Loan_ID and Loan_Status are excluded). This is copied directly from
# train_model.py's FEATURE_COLUMNS constant.
FEATURE_COLUMNS = [
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

REQUIRED_FIELDS = FEATURE_COLUMNS  # every feature is required on submit

# --------------------------------------------------------------------------- #
# Flask app + artifact loading (done once at startup, not per-request)
# --------------------------------------------------------------------------- #
app = Flask(__name__)

model = None
scaler = None
label_encoders = None
load_error = None  # populated if startup loading fails, surfaced in routes


def load_artifacts():
    """
    Load the trained model, scaler, and label encoders into memory.
    Any failure here is captured (not raised) so the app can still start
    and report a clear error on every request instead of crashing outright.
    """
    global model, scaler, label_encoders, load_error

    try:
        model = keras.models.load_model(MODEL_PATH)
    except Exception as exc:
        load_error = f"Failed to load model from '{MODEL_PATH}': {exc}"
        return

    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    except Exception as exc:
        load_error = f"Failed to load scaler from '{SCALER_PATH}': {exc}"
        return

    try:
        with open(ENCODERS_PATH, "rb") as f:
            label_encoders = pickle.load(f)
    except Exception as exc:
        load_error = f"Failed to load label encoders from '{ENCODERS_PATH}': {exc}"
        return


load_artifacts()


# --------------------------------------------------------------------------- #
# Preprocessing — mirrors train_model.py step for step
# --------------------------------------------------------------------------- #
def get_input_data():
    """
    Read the incoming request body regardless of whether it was sent as
    JSON (the index.html frontend uses `fetch()` with a JSON body) or as
    standard form-encoded data (e.g. curl, Postman, or a plain HTML form).

    Returns:
        dict of raw field values.

    Raises:
        ValueError: if no usable body was found.
    """
    if request.is_json:
        data = request.get_json(silent=True)
        if data is None:
            raise ValueError("Request body could not be parsed as JSON.")
        return dict(data)

    if request.form:
        return request.form.to_dict()

    # Some clients send JSON without a proper Content-Type header — try once
    # more before giving up.
    data = request.get_json(silent=True)
    if data:
        return dict(data)

    raise ValueError("No input data found in the request (expected JSON or form data).")


def preprocess_input(raw_data):
    """
    Convert raw request data into a single preprocessed feature row, using
    EXACTLY the same steps as train_model.py:
        1. Read + type-cast every field (raises ValueError on bad input)
        2. Label-encode categorical fields with the saved encoders
        3. Scale numerical fields with the saved StandardScaler
        4. Assemble the final vector in the training feature order

    Returns:
        np.ndarray of shape (1, len(FEATURE_COLUMNS)), ready for
        model.predict().

    Raises:
        ValueError: for missing fields, bad numeric input, or a category
                    value the encoder has never seen.
    """
    # --- 1. Collect + validate categorical values --------------------------
    raw = {}
    for col in CATEGORICAL_COLUMNS:
        value = raw_data.get(col, "")
        value = "" if value is None else str(value).strip()
        if value == "":
            raise ValueError(f"Missing value for '{col}'.")
        raw[col] = value

    # --- 2. Collect + validate numeric values -------------------------------
    numeric_fields = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]
    for col in numeric_fields:
        value = raw_data.get(col, None)
        if value is None or value == "":
            raise ValueError(f"Missing value for '{col}'.")
        try:
            raw[col] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{col}' must be a valid number (got '{value}').")

    # --- 3. Sanity-check numeric ranges -------------------------------------
    if raw["ApplicantIncome"] < 0 or raw["CoapplicantIncome"] < 0:
        raise ValueError("Income values cannot be negative.")
    if raw["LoanAmount"] <= 0:
        raise ValueError("Loan amount must be greater than 0.")
    if raw["Loan_Amount_Term"] <= 0:
        raise ValueError("Loan term must be greater than 0.")
    if raw["Credit_History"] not in (0.0, 1.0):
        raise ValueError("Credit history must be either 0 (poor) or 1 (good).")

    # --- 4. Label-encode categorical fields ---------------------------------
    encoded = dict(raw)
    for col in CATEGORICAL_COLUMNS:
        encoder = label_encoders.get(col)
        if encoder is None:
            raise ValueError(f"No encoder available for '{col}'.")
        try:
            encoded[col] = int(encoder.transform([raw[col]])[0])
        except ValueError:
            valid = ", ".join(encoder.classes_)
            raise ValueError(
                f"Unrecognized value '{raw[col]}' for '{col}'. Valid options: {valid}."
            )

    # --- 5. Assemble the feature row in the exact training column order ----
    row = [encoded[col] for col in FEATURE_COLUMNS]
    row_array = np.array(row, dtype="float32").reshape(1, -1)

    # --- 6. Scale only the numerical columns (same as training) ------------
    num_indices = [FEATURE_COLUMNS.index(c) for c in NUMERICAL_COLUMNS]
    numeric_slice = row_array[:, num_indices]
    row_array[:, num_indices] = scaler.transform(numeric_slice)

    return row_array


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/", methods=["GET"])
def home():
    """Render the loan application form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle a prediction request from index.html's JavaScript `fetch()` call:
      - read the JSON (or form) body
      - preprocess it exactly like training
      - run the neural network
      - return the result as JSON

    Response shape on success (HTTP 200):
        {
            "prediction": "Loan Approved" | "Loan Rejected",
            "approved": true | false,
            "probability": 0.0-1.0,      # raw model output, P(approved)
            "confidence": 0-100          # confidence % in the predicted class
        }

    Response shape on error (HTTP 400 for bad input, 500 for server errors):
        { "error": "<human-readable message>" }
    """
    # Guard against artifacts having failed to load at startup
    if load_error or model is None or scaler is None or label_encoders is None:
        return jsonify({"error": load_error or "Model artifacts are not loaded."}), 500

    try:
        raw_data = get_input_data()
        features = preprocess_input(raw_data)

        probability = float(model.predict(features, verbose=0)[0][0])
        approved = probability >= 0.5
        confidence = probability if approved else (1 - probability)

        return jsonify(
            {
                "prediction": "Loan Approved" if approved else "Loan Rejected",
                "approved": approved,
                "probability": round(probability, 4),
                "confidence": round(confidence * 100, 2),
            }
        ), 200

    except ValueError as ve:
        # Expected, user-facing input problems
        return jsonify({"error": str(ve)}), 400

    except Exception:
        # Unexpected server-side problems — log full traceback, show a
        # generic message to the client (avoid leaking internals).
        traceback.print_exc()
        return jsonify(
            {"error": "Something went wrong while making the prediction. Please try again."}
        ), 500


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(500)
def server_error(_):
    return jsonify({"error": "Internal server error."}), 500


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # debug=True is convenient for local development; set to False in production.
    app.run(debug=True)
