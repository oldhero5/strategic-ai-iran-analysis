#!/usr/bin/env python3
"""
Launch the beautiful D3.js Game Theory interface
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🎮 Launching Beautiful D3.js Game Theory Interface")
    print("=" * 60)
    
    # Change to the correct directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("📦 Dependencies already installed with uv...")
    # Dependencies should already be installed with uv sync
    
    print("🚀 Starting FastAPI server with beautiful D3.js interface...")
    print("   Access at: http://localhost:8000")
    print("   Press Ctrl+C to stop")
    print()
    
    try:
        # Run the FastAPI server
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "backend.api.server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--reload-dir", "frontend"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())