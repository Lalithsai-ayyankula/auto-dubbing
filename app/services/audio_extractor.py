"""
Audio Extraction Service
Step 1: Extract audio track from the input video using ffmpeg
"""

import os
import logging
import subprocess

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_dir: str, job_id: str) -> str:
    """
    Extract audio from a video file using ffmpeg.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the extracted audio
        job_id: Unique job identifier for naming

    Returns:
        Path to the extracted WAV audio file
    """
    audio_path = os.path.join(output_dir, f"{job_id}_original_audio.wav")

    logger.info(f"[{job_id}] Extracting audio from: {video_path}")

    command = [
        "ffmpeg",
        "-y",                      # Overwrite output if exists
        "-i", video_path,          # Input video
        "-vn",                     # No video output
        "-acodec", "pcm_s16le",    # 16-bit PCM WAV
        "-ar", "16000",            # 16kHz sample rate (required by Whisper)
        "-ac", "1",                # Mono channel
        audio_path
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # 5-minute timeout
        )
        if result.returncode != 0:
            error_msg = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed: {error_msg}")

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise RuntimeError("ffmpeg produced empty or missing audio file")

        logger.info(f"[{job_id}] Audio extracted: {audio_path}")
        return audio_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg audio extraction timed out after 5 minutes")
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install with: sudo apt install ffmpeg  "
            "or  brew install ffmpeg"
        )
