#!/bin/bash

# 1. Start FastAPI in the background
# We use 127.0.0.1 to be safe inside the container
uvicorn main:app --host 127.0.0.1 --port 8000 &

# 2. Start Streamlit in the foreground
# We disable CORS and XSRF protection to fix the 403 Forbidden error
streamlit run frontend.py --server.port 7860 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false