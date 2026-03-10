"""
Dubbing Router
Handles video upload, job management, and status polling endpoints
"""

import os
import uuid
import logging
import asyncio
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import aiofiles

from app.services.pipeline import DubbingPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory job store (use Redis for production)
jobs: dict = {}

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 500


@router.post("/dub")
async def start_dubbing(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Hindi educational video to dub"),
    language: str = Form(..., description="Target language: 'konkani' or 'maithili'"),
):
    """
    Upload a Hindi video and start the dubbing pipeline.
    Returns a job_id to track progress.
    """
    # Validate language
    if language not in ("konkani", "maithili"):
        raise HTTPException(
            status_code=400,
            detail="language must be 'konkani' or 'maithili'"
        )

    # Validate file extension
    ext = os.path.splitext(video.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Generate unique job ID
    job_id = str(uuid.uuid4())
    upload_path = os.path.join("uploads", f"{job_id}{ext}")

    # Save uploaded file to disk
    try:
        async with aiofiles.open(upload_path, "wb") as out_file:
            total_bytes = 0
            while chunk := await video.read(1024 * 1024):  # 1MB chunks
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
                    await out_file.close()
                    os.remove(upload_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Max size: {MAX_FILE_SIZE_MB}MB"
                    )
                await out_file.write(chunk)
        logger.info(f"[{job_id}] Saved upload: {upload_path} ({total_bytes / 1e6:.1f} MB)")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{job_id}] Failed to save upload")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Initialize job record
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "step": "Queued",
        "language": language,
        "input_path": upload_path,
        "output_path": None,
        "error": None,
    }

    # Run pipeline in background
    background_tasks.add_task(run_pipeline, job_id, upload_path, language)

    return {"job_id": job_id, "status": "queued", "message": "Dubbing started"}


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    """Poll job status and progress"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@router.get("/download/{job_id}")
async def download_result(job_id: str):
    """Get download URL for the completed dubbed video"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job['status']}"
        )

    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    filename = os.path.basename(output_path)
    return {"download_url": f"/outputs/{filename}", "filename": filename}


async def run_pipeline(job_id: str, video_path: str, language: str):
    """
    Run the full dubbing pipeline in a background task.
    Updates job status at each stage.
    """
    def update_job(status: str, progress: int, step: str, **kwargs):
        jobs[job_id].update({
            "status": status,
            "progress": progress,
            "step": step,
            **kwargs
        })
        logger.info(f"[{job_id}] [{progress}%] {step}")

    try:
        pipeline = DubbingPipeline(job_id=job_id, progress_callback=update_job)
        output_path = await asyncio.to_thread(pipeline.run, video_path, language)

        update_job(
            status="completed",
            progress=100,
            step="Done! Video ready for download.",
            output_path=output_path
        )

    except Exception as e:
        logger.exception(f"[{job_id}] Pipeline failed")
        jobs[job_id].update({
            "status": "failed",
            "step": f"Error: {str(e)}",
            "error": str(e),
        })
