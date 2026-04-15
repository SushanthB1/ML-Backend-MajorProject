# Use an official lightweight Python image (Debian-based slim is best for ML libraries)
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables to prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies (required for building certain ML/DB Python packages)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend source code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Default command (this gets overridden by docker-compose for live-reloading)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]