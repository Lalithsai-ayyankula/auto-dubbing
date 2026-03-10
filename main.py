"""
Automatic Dubbing for Educational Videos
Main FastAPI application entry point
"""

import os
import logging
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.routers import dubbing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure required directories exist
for directory in ["uploads", "outputs", "models", "static"]:
    os.makedirs(directory, exist_ok=True)

# Initialize FastAPI application
app = FastAPI(
    title="Auto Dubbing API",
    description="Automatic dubbing of Hindi educational videos to Konkani/Maithili",
    version="1.0.0"
)

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(dubbing.router, prefix="/api", tags=["dubbing"])


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main frontend page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Auto Dubbing API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, loop="asyncio")