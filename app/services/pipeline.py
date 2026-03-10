"""
Dubbing Pipeline Orchestrator
Coordinates all stages of the dubbing process:
  Step 1: Audio extraction (ffmpeg)
  Step 2: Transcription (Whisper)
  Step 3: Translation (mT5)
  Step 4: Speech synthesis (Coqui TTS)
  Step 5: Lip sync (Wav2Lip)
  Step 6: Return final video path
"""

import os
import json
import logging
from typing import Callable, Optional

from app.services.audio_extractor import extract_audio
from app.services.transcriber import transcribe_audio
from app.services.translator import translate_segments
from app.services.tts_service import synthesize_segments
from app.services.lip_sync import apply_lip_sync

logger = logging.getLogger(__name__)

OUTPUTS_DIR = "outputs"
TEMP_DIR = os.path.join(OUTPUTS_DIR, "temp")


class DubbingPipeline:
    """
    Orchestrates the full video dubbing pipeline.
    Calls progress_callback at each stage with (status, progress, step) args.
    """

    def __init__(self, job_id: str, progress_callback: Optional[Callable] = None):
        self.job_id = job_id
        self._cb = progress_callback or (lambda **kw: None)
        self.temp_dir = os.path.join(TEMP_DIR, job_id)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(OUTPUTS_DIR, exist_ok=True)

    def _progress(self, status: str, progress: int, step: str, **extra):
        self._cb(status=status, progress=progress, step=step, **extra)

    def run(self, video_path: str, language: str) -> str:
        """
        Execute the full dubbing pipeline.

        Args:
            video_path: Absolute or relative path to uploaded video
            language: 'konkani' or 'maithili'

        Returns:
            Path to the final dubbed video file
        """
        jid = self.job_id
        logger.info(f"[{jid}] Pipeline starting. Language={language}, Video={video_path}")

        # ── Step 1: Extract Audio ────────────────────────────────────────────
        self._progress("running", 10, "Step 1/5: Extracting audio from video...")
        audio_path = extract_audio(video_path, self.temp_dir, jid)
        logger.info(f"[{jid}] Step 1 complete: {audio_path}")

        # ── Step 2: Transcribe with Whisper ──────────────────────────────────
        self._progress("running", 25, "Step 2/5: Transcribing Hindi speech with Whisper...")
        segments = transcribe_audio(audio_path, jid)
        logger.info(f"[{jid}] Step 2 complete: {len(segments)} segments")

        # Save transcript for debugging/review
        transcript_path = os.path.join(self.temp_dir, f"{jid}_transcript.json")
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        # ── Step 3: Translate with mT5 ───────────────────────────────────────
        self._progress("running", 45, f"Step 3/5: Translating to {language.title()} with mT5...")
        translated = translate_segments(segments, language, jid)
        logger.info(f"[{jid}] Step 3 complete: {len(translated)} translated segments")

        # Save translation for debugging/review
        translation_path = os.path.join(self.temp_dir, f"{jid}_translation.json")
        with open(translation_path, "w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False, indent=2)

        # ── Step 4: Synthesize Speech with Coqui TTS ─────────────────────────
        self._progress("running", 60, "Step 4/5: Generating dubbed speech with Coqui TTS...")
        synth_audio_path = synthesize_segments(
            segments=translated,
            language=language,
            reference_audio_path=audio_path,   # For voice cloning
            output_dir=self.temp_dir,
            job_id=jid,
        )
        logger.info(f"[{jid}] Step 4 complete: {synth_audio_path}")

        # ── Step 5: Lip Sync with Wav2Lip ────────────────────────────────────
        self._progress("running", 80, "Step 5/5: Applying lip sync with Wav2Lip...")
        output_filename = f"{jid}_{language}_dubbed.mp4"
        output_path = os.path.join(OUTPUTS_DIR, output_filename)

        final_path = apply_lip_sync(
            video_path=video_path,
            audio_path=synth_audio_path,
            output_path=output_path,
            job_id=jid,
        )
        logger.info(f"[{jid}] Step 5 complete: {final_path}")

        # ── Cleanup temp files ───────────────────────────────────────────────
        self._cleanup_temp()

        logger.info(f"[{jid}] Pipeline complete! Output: {final_path}")
        return final_path

    def _cleanup_temp(self):
        """Remove intermediate temp files to save disk space."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.debug(f"[{self.job_id}] Temp directory cleaned: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"[{self.job_id}] Temp cleanup failed: {e}")
