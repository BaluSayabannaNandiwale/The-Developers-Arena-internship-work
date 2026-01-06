"""
Main entry point to run the Real Estate Price Prediction System
Starts both backend and frontend services
"""
import subprocess
import sys
import time
import threading
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("=" * 60)
    print("🏠 Real Estate Price Prediction System")
    print("=" * 60)
    print("\nStarting services...\n")

def start_backend():
    """Start backend server in a separate thread"""
    try:
        import uvicorn
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        uvicorn.run(
            "backend.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Backend error: {e}")

def start_frontend():
    """Start frontend server"""
    # Wait for backend to be ready
    time.sleep(3)
    
    try:
        project_root = Path(__file__).parent
        frontend_path = project_root / "frontend" / "app.py"
        
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(frontend_path),
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true"
        ])
    except Exception as e:
        print(f"❌ Frontend error: {e}")

if __name__ == "__main__":
    print_banner()
    
    print("📍 Services will be available at:")
    print("   • Backend API: http://localhost:8000")
    print("   • Frontend UI: http://localhost:8501")
    print("   • API Docs: http://localhost:8000/docs")
    print("\n⏳ Starting services...\n")
    print("Press Ctrl+C to stop all services\n")
    
    # Start backend in a daemon thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Start frontend in main thread (blocking)
    try:
        start_frontend()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
        print("✅ Services stopped. Goodbye!")

