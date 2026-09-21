# Loan Approval Prediction Using Neural Networks

A machine learning web application that predicts whether a loan application is likely to be **approved** or **rejected**, using a neural network trained on applicant and loan attributes. The project covers the full pipeline: synthetic dataset, data preprocessing, model training with TensorFlow/Keras, and a Flask web interface for interactive predictions.

> **Disclaimer:** This project is an educational/demonstration tool. It does not represent a real lending decision, is not connected to any financial institution, and should not be used to make actual credit decisions.

---

## 1. Project Purpose

Manual loan underwriting relies on evaluating an applicant's income, credit history, employment status, and other factors to judge repayment risk. This project demonstrates how a **feed-forward neural network** can learn patterns from historical loan data and estimate the likelihood of approval for a new application.

The neural network:
- Takes in a set of applicant and loan attributes (see below).
- Learns non-linear relationships between those attributes and the historical approval outcome during training.
- Outputs a single probability (0–1) via a sigmoid activation, representing the model's confidence that the loan should be approved.
- That probability is thresholded at 0.5 to produce a final **Approved / Rejected** label, and is also shown to the user as a confidence percentage.

---

## 2. Dataset & Input Features

The model is trained on `loan_approval_dataset.csv`, a tabular dataset where each row represents one loan application.

### Input Features

| Feature | Type | Description |
|---|---|---|
| `Gender` | Categorical | Applicant's gender (`Male` / `Female`) |
| `Married` | Categorical | Applicant's marital status (`Yes` / `No`) |
| `Dependents` | Categorical | Number of dependents (`0`, `1`, `2`, `3+`) |
| `Education` | Categorical | Applicant's education level (`Graduate` / `Not Graduate`) |
| `Self_Employed` | Categorical | Whether the applicant is self-employed (`Yes` / `No`) |
| `ApplicantIncome` | Numerical | Applicant's monthly income |
| `CoapplicantIncome` | Numerical | Co-applicant's monthly income (0 if none) |
| `LoanAmount` | Numerical | Requested loan amount (in thousands) |
| `Loan_Amount_Term` | Numerical | Loan repayment term, in months |
| `Credit_History` | Numerical (0/1) | Whether the applicant meets credit guidelines (`1` = good, `0` = poor) |
| `Property_Area` | Categorical | Location type of the property (`Urban`, `Semiurban`, `Rural`) |

### Target

| Column | Description |
|---|---|
| `Loan_Status` | `Y` (approved) or `N` (rejected) — the label the model is trained to predict |

An identifier column, `Loan_ID`, is also present in the raw CSV but is dropped before training since it carries no predictive information.

---

## 3. Data Preprocessing & Model Training

Preprocessing and training are handled end-to-end by `train_model.py`.

### Preprocessing steps

1. **Load data** — reads `loan_approval_dataset.csv` and drops the `Loan_ID` column.
2. **Handle missing values**
   - Categorical columns are filled with the **mode** (most frequent value).
   - Numerical columns are filled with the **median**.
3. **Encode categorical features** — each categorical column (`Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`), along with the target `Loan_Status`, is transformed with scikit-learn's `LabelEncoder`. The fitted encoders are saved so the exact same mapping can be reused at prediction time.
4. **Train/test split** — the dataset is split **80% training / 20% testing**, stratified on the target so both sets preserve the original class balance.
5. **Feature scaling** — the numerical columns (`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`) are standardized with `StandardScaler`, fitted **only on the training set** to avoid data leakage, then applied to the test set.

### Model architecture

A simple, fully-connected feed-forward neural network built with TensorFlow/Keras:

```
Input layer
   → Dense(32, activation="relu")
   → Dense(16, activation="relu")
   → Dense(1,  activation="sigmoid")
```

- **Loss function:** binary cross-entropy
- **Optimizer:** Adam
- **Metric tracked during training:** accuracy
- **Epochs:** trains for up to 50 epochs, with an `EarlyStopping` callback (monitoring validation loss) that restores the best-performing weights if the model stops improving sooner.

### Evaluation

After training, the model is evaluated on the held-out 20% test set. The script prints:
- Accuracy, Precision, Recall, and F1-score
- A confusion matrix
- A full scikit-learn classification report

Exact metric values depend on the dataset used and will be printed to the console each time `train_model.py` is run — they are not fixed or hard-coded.

### Saved artifacts

Running `train_model.py` produces three files, all required by the Flask app:

| File | Contents |
|---|---|
| `loan_model.keras` | The trained neural network |
| `scaler.pkl` | The fitted `StandardScaler` for numerical features |
| `label_encoders.pkl` | A dictionary of fitted `LabelEncoder` objects (one per categorical column, plus the target) |

---

## 4. Flask Backend & Web Interface

`app.py` serves a small web application built around the trained model:

- **On startup**, it loads `loan_model.keras`, `scaler.pkl`, and `label_encoders.pkl` once into memory.
- **`GET /`** renders an HTML form (`templates/index.html`) where a user fills in the same fields the model was trained on. The dropdown options for categorical fields (e.g. `Property_Area`, `Education`) are generated directly from the saved label encoders, so the form can never drift out of sync with what the model actually understands.
- **`POST /predict`** receives the submitted form data and:
  1. Validates and type-casts every field (catching missing fields, invalid numbers, and out-of-range values).
  2. Applies the **same preprocessing used during training** — label encoding via the saved encoders, then scaling the numerical fields with the saved scaler — and assembles the feature vector in the exact column order the model expects.
  3. Runs the feature vector through the loaded neural network to get a probability.
  4. Converts that probability into a final label (**Loan Approved** / **Loan Rejected**) using a 0.5 threshold, and computes a confidence percentage.
  5. Re-renders the same page, showing the result alongside the originally submitted values.
- **Error handling:** invalid or missing input, unrecognized categorical values, and any unexpected server-side errors are all caught and surfaced to the user as a clear message instead of a raw stack trace or a crash. If the model/scaler/encoders fail to load at startup, every request reports that clearly rather than the app silently failing.

---

## 5. Project Features

- End-to-end pipeline: dataset → preprocessing → neural network training → web-based inference.
- Clean separation between training code (`train_model.py`) and serving code (`app.py`).
- Preprocessing consistency guaranteed between training and inference by reusing the same saved `scaler.pkl` and `label_encoders.pkl`.
- Interactive web form with input validation and human-readable error messages.
- Prediction result shown as both a **class label** (Approved/Rejected) and a **confidence percentage**.
- Form dropdowns generated dynamically from the trained encoders, avoiding hard-coded/out-of-sync category lists.
- Defensive error handling throughout the backend (missing fields, bad values, unseen categories, load failures).

---

## 6. Technologies & Libraries Used

| Category | Technology |
|---|---|
| Language | Python 3 |
| Deep Learning | TensorFlow / Keras |
| Data Handling | pandas, NumPy |
| Preprocessing & Metrics | scikit-learn (`LabelEncoder`, `StandardScaler`, `train_test_split`, evaluation metrics) |
| Web Framework | Flask |
| Frontend | HTML, CSS (Jinja2 templating via Flask) |
| Model/Object Persistence | Keras native format (`.keras`), Python `pickle` (`.pkl`) |

See `requirements.txt` for exact package versions.

---

## 7. Installation & Setup

### Prerequisites
- Python 3.9+ installed
- `pip` available on your PATH

### Steps

1. **Clone or download the project** into a local folder and open it in VS Code (or your editor of choice).

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Confirm the dataset is present** — `loan_approval_dataset.csv` should be in the project root alongside `train_model.py`.

---

## 8. How to Train the Model

From the project root, run:

```bash
python train_model.py
```

This will:
1. Load and preprocess `loan_approval_dataset.csv`.
2. Split the data into training and test sets (80/20).
3. Build and train the neural network for up to 50 epochs.
4. Print Accuracy, Precision, Recall, F1-score, and a confusion matrix to the console.
5. Save `loan_model.keras`, `scaler.pkl`, and `label_encoders.pkl` into the project root.

These three saved files are required before the Flask app can serve predictions — run this step first (or re-run it any time the dataset changes).

---

## 9. How to Run the Flask Application

Once `loan_model.keras`, `scaler.pkl`, and `label_encoders.pkl` exist in the project root:

```bash
python app.py
```

By default, Flask will start a local development server at:

```
http://127.0.0.1:5000/
```

Open that address in a browser to access the application form.

---

## 10. Project Folder Structure

```
loan-approval-prediction/
│
├── loan_approval_dataset.csv   # Training dataset
├── train_model.py              # Preprocessing + model training script
├── app.py                      # Flask backend (routes, inference, error handling)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── templates/
│   └── index.html              # Loan application form + result display
│
├── static/
│   └── style.css                # Styling for the web interface
│
├── loan_model.keras            # Trained neural network (generated by train_model.py)
├── scaler.pkl                  # Fitted StandardScaler (generated by train_model.py)
└── label_encoders.pkl          # Fitted LabelEncoders (generated by train_model.py)
```

---

## 11. How a User Makes a Loan Prediction

1. Start the Flask app (`python app.py`) and open `http://127.0.0.1:5000/` in a browser.
2. Fill in the application form with the applicant's details:
   - Gender, Married, Dependents, Education, Self-Employed status
   - Applicant Income and Coapplicant Income
   - Loan Amount and Loan Term
   - Credit History
   - Property Area
3. Click **Submit Application**.
4. The backend preprocesses the input exactly as it was preprocessed during training, runs it through the neural network, and returns:
   - A result label — **Loan Approved** or **Loan Rejected**
   - A **confidence percentage**, representing how confident the model is in that outcome
5. If any field is missing, invalid, or contains a value the model wasn't trained on, the form redisplays with a clear error message instead of a prediction, and the previously entered values are preserved so they don't need to be re-typed.

---

## License / Usage Note

This project was built for educational purposes (coursework, portfolio, and demonstration use). It is not intended for production lending decisions.
