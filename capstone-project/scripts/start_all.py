"""
Start Both Backend and Frontend
"""
import subprocess
import sys
import time
from pathlib import Path
import threading

def start_backend():
    """Start backend server"""
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

def start_frontend():
    """Start frontend server"""
    project_root = Path(__file__).parent.parent
    frontend_path = project_root / "frontend" / "app.py"
    
    # Wait a bit for backend to start
    time.sleep(3)
    
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(frontend_path),
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ])

if __name__ == "__main__":
    print("Starting Real Estate Price Prediction System...")
    print("Backend will be available at: http://localhost:8000")
    print("Frontend will be available at: http://localhost:8501")
    print("API Documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop all services\n")
    
    # Start backend in a thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Start frontend in main thread
    try:
        start_frontend()
    except KeyboardInterrupt:
        print("\nShutting down...")

