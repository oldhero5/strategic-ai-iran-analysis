#!/usr/bin/env python3
"""
Run the MCMC API server for the Game Theory Iran Model

This script starts the FastAPI server with MCMC endpoints for the frontend.
Run this alongside the main D3 application for full MCMC integration.
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the FastAPI app
from backend.api.mcmc_endpoints import app, initialize_models

def main():
    """Start the MCMC API server"""
    print("🚀 Starting Game Theory MCMC API Server...")
    print("📊 This provides MCMC/Bayesian endpoints for the frontend")
    print("🌐 Frontend should be running on http://localhost:8000")
    print("🔌 API will be available on http://localhost:8001")
    print("")
    
    # Ensure models are initialized
    initialize_models()
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()