# Multi-stage Docker build for the flight delay prediction API
FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

# Add user bin directory to PATH so installed scripts (e.g. pytest) are found
ENV PATH=/home/appuser/.local/bin:$PATH

# Set Python path to include project root
ENV PYTHONPATH=/app/src

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir uvicorn[standard]

# Copy only the source code needed for the API
COPY src/ ./src/

EXPOSE 8000

# Command to run the API
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]