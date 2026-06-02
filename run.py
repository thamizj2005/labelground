#!/usr/bin/env python3
"""
Offline Annotation Platform - Main Entry Point
Run with: python run.py
"""

import uvicorn
import sys
import socket
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def get_local_ip():
    """Get the machine's local network IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unable to detect"

if __name__ == "__main__":
    local_ip = get_local_ip()
    print("=" * 60)
    print("Starting Offline Annotation Platform")
    print("=" * 60)
    print(f"Access from this PC:      http://localhost:8000")
    print(f"Access from any device:   http://{local_ip}:8000")
    print(f"API documentation:        http://localhost:8000/docs")
    print("=" * 60)
    
    # Run the FastAPI app
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
