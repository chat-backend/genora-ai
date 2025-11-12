#!/bin/bash

# In thông báo khởi động
echo "🚀 Starting Genora-AI service with Uvicorn..."

# Khởi động ứng dụng FastAPI bằng Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
