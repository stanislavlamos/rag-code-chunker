import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def run_backend():
    """Run the FastAPI backend server"""
    backend_dir = Path("backend")
    subprocess.run([sys.executable, "-m", "uvicorn", "api.main:app", "--reload"], cwd=backend_dir)

def run_frontend():
    """Run the Streamlit frontend server"""
    frontend_dir = Path("frontend")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=frontend_dir)

if __name__ == "__main__":
    # Start backend in a separate process
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--reload"],
        cwd="backend"
    )
    
    # Wait a moment for backend to start
    time.sleep(2)
    
    # Start frontend in a separate process
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py"],
        cwd="frontend"
    )
    
    # Open the frontend in the default browser
    webbrowser.open("http://localhost:8501")
    
    try:
        # Wait for both processes
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nShutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        backend_process.wait()
        frontend_process.wait()
        print("Servers stopped.") 