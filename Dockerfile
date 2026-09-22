# ---------------------------------------------------------------------------
# Dockerfile — Loan Approval Prediction Using Neural Networks
# ---------------------------------------------------------------------------
# Builds a container that serves the existing Flask app (app.py) with the
# trained model and preprocessing artifacts baked in, run behind Gunicorn
# for a production-appropriate setup.
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# Prevent .pyc files and enable unbuffered logging (useful in containers)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set the working directory inside the container
WORKDIR /app

# Install OS-level dependencies required to build/run some Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (separate layer for better build caching)
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip install gunicorn

# Copy application code, templates, static assets, and trained artifacts.
# (loan_model.keras, scaler.pkl, and label_encoders.pkl must exist in the
# build context — generated beforehand by running train_model.py.)
COPY app.py .
COPY templates/ templates/
COPY loan_model.keras .
COPY scaler.pkl .
COPY label_encoders.pkl .

# Run as a non-root user for better container security
RUN useradd --create-home appuser
USER appuser

# Flask/Gunicorn will listen on this port
EXPOSE 5000

# Basic container-level health check against the home route
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/')" || exit 1

# Serve the existing Flask app object (app.py defines `app = Flask(__name__)`)
# with Gunicorn instead of the Flask development server.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
