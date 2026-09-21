"""
test_model.py
==============
Test suite for the "Loan Approval Prediction Using Neural Networks" project.

This script verifies, end-to-end, that the artifacts produced by
train_model.py (loan_model.keras, scaler.pkl, label_encoders.pkl) load
correctly and behave consistently with the preprocessing used in app.py:

  1. The trained Keras model loads without error.
  2. The saved StandardScaler loads without error.
  3. The saved LabelEncoders load without error.
  4. Sample loan applications can be run through the SAME preprocessing
     steps used at training/inference time (missing-value handling,
     label encoding, scaling, column ordering).
  5. The resulting feature vector has the exact shape the model expects.
  6. The model produces a valid prediction (a probability strictly
     between 0 and 1, and a well-formed Approved/Rejected label).
  7. Two contrasting applicant profiles (a strong / weak profile) are run
     through the model to exercise both the "Approved" and "Rejected"
     code paths where the model's own scoring allows it — this is a
     sanity check on the pipeline, not a claim about model accuracy.

This script does NOT report or assert any fixed accuracy/performance
number. It only reports PASS/FAIL for pipeline correctness.

Run:
    python test_model.py
"""

import os
import pickle
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Column configuration — MUST mirror train_model.py / app.py exactly, since
# the scaler / encoders / model were all fit using these definitions.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")

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

# Exact feature order the model was trained on (must match train_model.py).
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

EXPECTED_FEATURE_COUNT = len(FEATURE_COLUMNS)  # 11

# ---------------------------------------------------------------------------
# Sample applicant profiles used for testing.
#
# These are illustrative examples only — they are designed to bias the
# RAW INPUT toward "should look strong" / "should look weak" on paper
# (good credit history + comfortable income + small loan vs. the reverse),
# not to guarantee a specific model output. The model's own learned
# weights determine the actual prediction.
# ---------------------------------------------------------------------------
STRONG_PROFILE_RAW = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 9000.0,
    "CoapplicantIncome": 3000.0,
    "LoanAmount": 80.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1.0,
    "Property_Area": "Semiurban",
}

WEAK_PROFILE_RAW = {
    "Gender": "Female",
    "Married": "No",
    "Dependents": "3+",
    "Education": "Not Graduate",
    "Self_Employed": "Yes",
    "ApplicantIncome": 1200.0,
    "CoapplicantIncome": 0.0,
    "LoanAmount": 600.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 0.0,
    "Property_Area": "Rural",
}


# ---------------------------------------------------------------------------
# Small test-reporting helpers (no external test framework required)
# ---------------------------------------------------------------------------
class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def record(self, name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {name}"
        if detail:
            line += f" — {detail}"
        print(line)
        if ok:
            self.passed += 1
        else:
            self.failed += 1

    def summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 60)
        print(f"TEST SUMMARY: {self.passed}/{total} passed, {self.failed} failed")
        print("=" * 60)
        return self.failed == 0


results = TestResult()


# ---------------------------------------------------------------------------
# Preprocessing helper — mirrors the logic in app.py's
# preprocess_form_input(), applied to a plain dict instead of a Flask form.
# ---------------------------------------------------------------------------
def preprocess_sample(raw, scaler, label_encoders):
    """
    Convert a raw applicant dict into a preprocessed feature vector using
    the same steps as training/inference:
        1. Label-encode categorical fields with the saved encoders
        2. Scale numerical fields with the saved StandardScaler
        3. Assemble the row in the exact training feature order

    Returns:
        np.ndarray of shape (1, len(FEATURE_COLUMNS))
    """
    encoded = dict(raw)

    for col in CATEGORICAL_COLUMNS:
        encoder = label_encoders[col]
        encoded[col] = int(encoder.transform([str(raw[col])])[0])

    row = [encoded[col] for col in FEATURE_COLUMNS]
    row_array = np.array(row, dtype="float32").reshape(1, -1)

    num_indices = [FEATURE_COLUMNS.index(c) for c in NUMERICAL_COLUMNS]
    numeric_slice = row_array[:, num_indices]
    row_array[:, num_indices] = scaler.transform(numeric_slice)

    return row_array


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------
def test_model_loads():
    """The trained Keras model file should load without error."""
    try:
        from tensorflow import keras
        model = keras.models.load_model(MODEL_PATH)
        results.record(
            "Model loads correctly",
            True,
            f"loaded from '{os.path.basename(MODEL_PATH)}'",
        )
        return model
    except Exception as exc:
        results.record("Model loads correctly", False, str(exc))
        return None


def test_scaler_loads():
    """The saved StandardScaler should load and expose fitted attributes."""
    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        has_stats = hasattr(scaler, "mean_") and hasattr(scaler, "scale_")
        results.record(
            "Scaler loads correctly",
            has_stats,
            "fitted StandardScaler with mean_/scale_ present"
            if has_stats
            else "loaded object is missing fitted attributes (mean_/scale_)",
        )
        return scaler if has_stats else None
    except Exception as exc:
        results.record("Scaler loads correctly", False, str(exc))
        return None


def test_encoders_load():
    """The saved label encoders dict should load and contain all expected columns."""
    try:
        with open(ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)
        missing = [c for c in CATEGORICAL_COLUMNS if c not in encoders]
        ok = len(missing) == 0
        results.record(
            "Label encoders load correctly",
            ok,
            "all expected categorical encoders present"
            if ok
            else f"missing encoders for: {missing}",
        )
        return encoders if ok else None
    except Exception as exc:
        results.record("Label encoders load correctly", False, str(exc))
        return None


def test_preprocessing_pipeline(scaler, encoders):
    """Sample raw applicant data should preprocess into a numeric feature vector."""
    try:
        vector = preprocess_sample(STRONG_PROFILE_RAW, scaler, encoders)
        is_numeric = np.issubdtype(vector.dtype, np.floating)
        no_nans = not np.isnan(vector).any()
        ok = is_numeric and no_nans
        results.record(
            "Preprocessing pipeline runs on sample data",
            ok,
            f"produced array of dtype {vector.dtype}, shape {vector.shape}",
        )
        return vector if ok else None
    except Exception as exc:
        results.record("Preprocessing pipeline runs on sample data", False, str(exc))
        return None


def test_feature_vector_shape(vector):
    """The preprocessed feature vector must match the shape the model expects."""
    if vector is None:
        results.record("Feature vector has correct shape", False, "no vector produced by prior step")
        return
    ok = vector.shape == (1, EXPECTED_FEATURE_COUNT)
    results.record(
        "Feature vector has correct shape",
        ok,
        f"expected (1, {EXPECTED_FEATURE_COUNT}), got {vector.shape}",
    )


def test_model_accepts_input(model, vector):
    """The model should accept the preprocessed vector without a shape/type error."""
    if model is None or vector is None:
        results.record("Model accepts preprocessed input", False, "model or vector unavailable")
        return None
    try:
        output = model.predict(vector, verbose=0)
        ok = output.shape == (1, 1)
        results.record(
            "Model accepts preprocessed input",
            ok,
            f"output shape {output.shape}",
        )
        return output if ok else None
    except Exception as exc:
        results.record("Model accepts preprocessed input", False, str(exc))
        return None


def test_valid_probability(output):
    """The model's raw output should be a single valid probability in [0, 1]."""
    if output is None:
        results.record("Model produces a valid probability", False, "no output available")
        return
    prob = float(output[0][0])
    ok = 0.0 <= prob <= 1.0 and not np.isnan(prob)
    results.record(
        "Model produces a valid probability",
        ok,
        f"probability = {prob:.4f}",
    )


def test_approved_and_rejected_paths(model, scaler, encoders):
    """
    Run a 'strong' and a 'weak' applicant profile through the full pipeline
    and confirm both produce valid, well-formed predictions — exercising
    both the Approved and Rejected branches of the decision logic.

    Note: this checks that the PIPELINE correctly produces both possible
    labels and that the model responds differently to different inputs.
    It does not assert or report a fixed accuracy value.
    """
    try:
        strong_vec = preprocess_sample(STRONG_PROFILE_RAW, scaler, encoders)
        weak_vec = preprocess_sample(WEAK_PROFILE_RAW, scaler, encoders)

        strong_prob = float(model.predict(strong_vec, verbose=0)[0][0])
        weak_prob = float(model.predict(weak_vec, verbose=0)[0][0])

        strong_label = "Loan Approved" if strong_prob >= 0.5 else "Loan Rejected"
        weak_label = "Loan Approved" if weak_prob >= 0.5 else "Loan Rejected"

        print(f"      Strong profile -> {strong_label} (probability={strong_prob:.4f})")
        print(f"      Weak profile   -> {weak_label} (probability={weak_prob:.4f})")

        both_valid = (0.0 <= strong_prob <= 1.0) and (0.0 <= weak_prob <= 1.0)
        results.record(
            "Both profiles produce valid predictions",
            both_valid,
            "each profile returned a well-formed probability and label",
        )

        # Directional sanity check: the profile with better credit history,
        # higher income, and a smaller loan should score at least as high
        # as the weaker profile. This is a soft pipeline sanity check, not
        # a claim about the model's overall accuracy.
        directionally_sound = strong_prob >= weak_prob
        results.record(
            "Model differentiates strong vs. weak profile",
            directionally_sound,
            f"strong={strong_prob:.4f} vs weak={weak_prob:.4f}",
        )

        distinct_labels = strong_label != weak_label
        if distinct_labels:
            results.record(
                "Both Approved and Rejected labels observed",
                True,
                f"strong -> {strong_label}, weak -> {weak_label}",
            )
        else:
            # Not a failure — with a given trained model this can happen —
            # but it's surfaced clearly rather than silently skipped.
            results.record(
                "Both Approved and Rejected labels observed",
                False,
                f"both profiles resolved to '{strong_label}'; try more extreme "
                f"sample profiles to observe both outcomes with this model",
            )

    except Exception as exc:
        results.record("Both profiles produce valid predictions", False, str(exc))


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("LOAN APPROVAL PREDICTION — MODEL & PIPELINE TEST SUITE")
    print("=" * 60)

    # 1-3: artifact loading
    model = test_model_loads()
    scaler = test_scaler_loads()
    encoders = test_encoders_load()

    if model is None or scaler is None or encoders is None:
        print("\nOne or more required artifacts failed to load.")
        print("Run 'python train_model.py' first to generate them, then re-run this test.")
        results.summary()
        sys.exit(1)

    # 4-5: preprocessing + shape
    vector = test_preprocessing_pipeline(scaler, encoders)
    test_feature_vector_shape(vector)

    # 6: model accepts input / produces valid probability
    output = test_model_accepts_input(model, vector)
    test_valid_probability(output)

    # 7: approved vs. rejected paths
    print("\nRunning contrasting applicant profiles through the full pipeline...")
    test_approved_and_rejected_paths(model, scaler, encoders)

    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
