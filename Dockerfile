FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install required packages and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy the app code into the container
COPY . /app/

# Remove any secrets that might have been copied and create directory for mounted secrets
RUN rm -rf /app/etc/secrets/ && mkdir -p /app/etc/secrets

# Expose port 8000 (Uvicorn will run here)
EXPOSE 8000

# Run the Uvicorn server
# This path matches your actual project structure
CMD ["gunicorn", "--config", "gunicorn_config.py", "com.mhire.app.main:app"]