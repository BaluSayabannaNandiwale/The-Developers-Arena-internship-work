"""
Start Frontend Server
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    frontend_path = project_root / "frontend" / "app.py"
    
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(frontend_path),
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ])

