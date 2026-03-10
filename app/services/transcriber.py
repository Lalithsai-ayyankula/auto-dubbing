"""
Whisper Transcription Service
Step 2: Transcribe Hindi speech from extracted audio using OpenAI Whisper
"""

import logging
import torch

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading on every request
_whisper_model = None


def get_whisper_model(model_size: str = "small"):
    """
    Load Whisper model, caching it globally for reuse.
    Falls back to CPU if no GPU is available.
    """
    global _whisper_model
    if _whisper_model is None:
        import whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper '{model_size}' model on {device}...")
        _whisper_model = whisper.load_model(model_size, device=device)
        logger.info("Whisper model loaded successfully")
    return _whisper_model


def transcribe_audio(audio_path: str, job_id: str) -> list[dict]:
    """
    Transcribe Hindi audio using Whisper.

    Args:
        audio_path: Path to the WAV audio file
        job_id: Job identifier for logging

    Returns:
        List of segment dicts with keys: 'start', 'end', 'text'
        Example: [{'start': 0.0, 'end': 2.5, 'text': 'नमस्ते'}, ...]
    """
    logger.info(f"[{job_id}] Starting Whisper transcription: {audio_path}")
    model = get_whisper_model("small")

    # Transcribe with Hindi language hint for better accuracy
    result = model.transcribe(
        audio_path,
        language="hi",           # Force Hindi for educational videos
        task="transcribe",       # Transcribe (not translate)
        verbose=False,
        fp16=torch.cuda.is_available(),
        word_timestamps=False,
        condition_on_previous_text=True,
    )

    segments = result.get("segments", [])
    logger.info(f"[{job_id}] Transcription complete. Segments: {len(segments)}")

    # Normalize segment structure
    normalized = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:  # Skip empty segments
            normalized.append({
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": text,
            })

    if not normalized:
        raise RuntimeError("Whisper returned no transcription segments. Check audio quality.")

    logger.info(f"[{job_id}] Full transcript preview: {normalized[0]['text'][:80]}...")
    return normalized
