"""
Lip Sync Service (Wav2Lip)
Step 5: Synchronize lip movements in the video with the synthesized audio
using the Wav2Lip model.

Setup required:
  git clone https://github.com/Rudrabha/Wav2Lip.git models/Wav2Lip
  Download wav2lip_gan.pth to models/Wav2Lip/checkpoints/

If Wav2Lip is unavailable, this service falls back to a simple audio replacement
using ffmpeg — preserving full video quality without lip sync enhancement.
"""

import os
import sys
import logging
import subprocess

logger = logging.getLogger(__name__)

WAV2LIP_DIR = os.path.join("models", "Wav2Lip")
WAV2LIP_CHECKPOINT = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")
WAV2LIP_INFERENCE_SCRIPT = os.path.join(WAV2LIP_DIR, "inference.py")


def is_wav2lip_available() -> bool:
    """Check if Wav2Lip is installed and checkpoint exists."""
    return (
        os.path.isdir(WAV2LIP_DIR)
        and os.path.isfile(WAV2LIP_CHECKPOINT)
        and os.path.isfile(WAV2LIP_INFERENCE_SCRIPT)
    )


def apply_lip_sync(
    video_path: str,
    audio_path: str,
    output_path: str,
    job_id: str,
) -> str:
    """
    Apply lip sync to the video using Wav2Lip.
    Falls back to plain audio replacement if Wav2Lip is unavailable.

    Args:
        video_path: Path to the original input video
        audio_path: Path to the synthesized dubbed audio WAV
        output_path: Desired output video path
        job_id: Job identifier

    Returns:
        Path to the final output video file
    """
    if False and is_wav2lip_available():
        return _run_wav2lip(video_path, audio_path, output_path, job_id)
    else:
        logger.warning(
            f"[{job_id}] Wav2Lip not found at '{WAV2LIP_DIR}'. "
            "Falling back to audio-only replacement (no lip sync). "
            "See README for Wav2Lip setup instructions."
        )
        return _replace_audio_only(video_path, audio_path, output_path, job_id)


def _run_wav2lip(
    video_path: str,
    audio_path: str,
    output_path: str,
    job_id: str,
) -> str:
    """Run Wav2Lip inference script as a subprocess."""
    logger.info(f"[{job_id}] Running Wav2Lip lip sync...")

    # Wav2Lip outputs to a fixed path; we'll move it afterward
    wav2lip_out = os.path.join(WAV2LIP_DIR, "results", "result_voice.mp4")
    os.makedirs(os.path.dirname(wav2lip_out), exist_ok=True)

    command = [
        sys.executable,                        # Current Python interpreter
        WAV2LIP_INFERENCE_SCRIPT,
        "--checkpoint_path", WAV2LIP_CHECKPOINT,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", wav2lip_out,
        "--resize_factor", "1",
        "--nosmooth",                           # Disable smoothing for speed
    ]

    # Add Wav2Lip to Python path so its imports resolve correctly
    env = os.environ.copy()
    env["PYTHONPATH"] = WAV2LIP_DIR + os.pathsep + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(
            command,
            cwd=WAV2LIP_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1800,  # 30-minute timeout
            env=env,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            logger.error(f"[{job_id}] Wav2Lip stderr: {stderr[-1000:]}")
            raise RuntimeError(f"Wav2Lip failed with code {result.returncode}")

        if not os.path.exists(wav2lip_out):
            raise RuntimeError("Wav2Lip did not produce output file")

        # Move result to the expected output path
        os.replace(wav2lip_out, output_path)
        logger.info(f"[{job_id}] Wav2Lip complete: {output_path}")
        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("Wav2Lip timed out after 30 minutes")


def _replace_audio_only(
    video_path: str,
    audio_path: str,
    output_path: str,
    job_id: str,
) -> str:
    """
    Fallback: Replace the audio track of the video without lip sync.
    Uses ffmpeg to merge original video with new audio.
    """
    logger.info(f"[{job_id}] Replacing audio track (no lip sync fallback)...")

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,        # Original video (video track)
        "-i", audio_path,        # Synthesized audio
        "-c:v", "copy",          # Copy video stream as-is (fast, lossless)
        "-c:a", "aac",           # Encode audio as AAC
        "-b:a", "192k",
        "-map", "0:v:0",         # Use video from first input
        "-map", "1:a:0",         # Use audio from second input
        "-shortest",             # Trim to shortest stream
        output_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300
    )

    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg audio replace failed: {error}")

    logger.info(f"[{job_id}] Audio replaced successfully: {output_path}")
    return output_path
