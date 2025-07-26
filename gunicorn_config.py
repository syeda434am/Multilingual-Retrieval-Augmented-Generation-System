# gunicorn_config.py
bind = "0.0.0.0:8000"
workers = 2  # Reduce workers to save memory
worker_class = "uvicorn.workers.UvicornWorker"

# Timeout settings for long-running operations
timeout = 10000  # 5 minutes for worker timeout
keepalive = 5
max_requests = 100
max_requests_jitter = 10

# Memory and connection settings
worker_connections = 1000
preload_app = True