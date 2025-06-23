"""
FastAPI server for the beautiful D3.js Game Theory frontend
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

app = FastAPI(
    title="Game Theory Iran Model",
    description="Beautiful D3.js visualizations for Iran-Israel-US strategic analysis",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "frontend" / "static"), name="static")

# Setup templates
templates = Jinja2Templates(directory=PROJECT_ROOT / "frontend" / "templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main D3.js interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Game Theory Model API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_dirs=[str(PROJECT_ROOT / "frontend")]
    )