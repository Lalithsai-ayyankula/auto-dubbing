"""
Text-to-Speech Service - Fixed for CPU with YourTTS speaker support
"""

import os
import logging
import numpy as np
import torch
import soundfile as sf

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_GPU = DEVICE == "cuda"

_tts_model    = None
_tts_speaker  = None
_tts_language = None


def get_tts_model():
    """Load TTS model and pick first available speaker/language."""
    global _tts_model, _tts_speaker, _tts_language

    if _tts_model is None:
        from TTS.api import TTS

        if IS_GPU:
            logger.info("GPU: Loading XTTS-v2 with voice cloning...")
            _tts_model    = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
            _tts_speaker  = None   # Uses speaker_wav for cloning
            _tts_language = "hi"
        else:
            logger.info("CPU: Loading YourTTS...")
            _tts_model = TTS("tts_models/multilingual/multi-dataset/your_tts")

            # Auto-select first available speaker and English language
            _tts_speaker  = _tts_model.speakers[0] if _tts_model.speakers else None
            _tts_language = "en-us" if "en-us" in (_tts_model.languages or []) else \
                            "en"    if "en"    in (_tts_model.languages or []) else \
                            (_tts_model.languages[0] if _tts_model.languages else None)

            logger.info(f"YourTTS speaker='{_tts_speaker}' language='{_tts_language}'")

    return _tts_model, _tts_speaker, _tts_language


def transliterate_to_latin(text: str) -> str:
    """Convert Devanagari to Latin so English TTS can pronounce it."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    except ImportError:
        pass
    try:
        from aksharamukha import transliterate as t
        return t.process('Devanagari', 'IAST', text)
    except ImportError:
        pass
    # If no transliteration available, return text as-is
    return text


def synthesize_segments(
    segments: list,
    language: str,
    reference_audio_path: str,
    output_dir: str,
    job_id: str,
) -> str:
    """Synthesize all translated segments into a single audio WAV file."""
    logger.info(f"[{job_id}] Synthesizing {len(segments)} segments on {DEVICE}")
    tts, speaker, tts_lang = get_tts_model()

    sample_rate   = 16000
    total_dur     = max(seg["end"] for seg in segments)
    total_samples = int(total_dur * sample_rate) + sample_rate
    full_audio    = np.zeros(total_samples, dtype=np.float32)

    total = len(segments)
    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        if not text:
            continue

        # Transliterate Devanagari → Latin for TTS pronunciation
        tts_text = transliterate_to_latin(text)
        logger.info(f"[{job_id}] Seg {i+1}/{total}: '{tts_text[:70]}'")

        seg_path    = os.path.join(output_dir, f"{job_id}_seg_{i:04d}.wav")
        start_smpl  = int(seg["start"] * sample_rate)
        allowed_dur = seg["end"] - seg["start"]

        try:
            if IS_GPU:
                # GPU: XTTS-v2 with voice cloning
                tts.tts_to_file(
                    text=tts_text,
                    speaker_wav=reference_audio_path,
                    language="hi",
                    file_path=seg_path,
                    split_sentences=False,
                )
            else:
                # CPU: YourTTS with explicit speaker + language
                kwargs = {"text": tts_text, "file_path": seg_path}
                if speaker:
                    kwargs["speaker"] = speaker
                if tts_lang:
                    kwargs["language"] = tts_lang
                tts.tts_to_file(**kwargs)

            # Load synthesized audio
            synth, sr = sf.read(seg_path)
            if synth.ndim > 1:
                synth = synth.mean(axis=1)

            # Check for silence
            if np.abs(synth).max() < 0.001:
                logger.warning(f"[{job_id}] Seg {i} is silent, skipping")
                continue

            # Resample to match our buffer rate
            if sr != sample_rate:
                import librosa
                synth = librosa.resample(synth, orig_sr=sr, target_sr=sample_rate)

            # Time-stretch to fit original segment duration
            synth_dur = len(synth) / sample_rate
            if abs(synth_dur - allowed_dur) > 0.3 and allowed_dur > 0.1:
                import librosa
                rate  = max(0.5, min(synth_dur / allowed_dur, 2.0))
                synth = librosa.effects.time_stretch(synth, rate=rate)

            # Place into full audio buffer at correct timestamp
            max_smpl = int(allowed_dur * sample_rate)
            synth    = synth[:max_smpl]
            end_smpl = start_smpl + len(synth)
            if end_smpl > len(full_audio):
                full_audio = np.pad(full_audio, (0, end_smpl - len(full_audio)))
            full_audio[start_smpl:end_smpl] += synth

        except Exception as e:
            logger.warning(f"[{job_id}] TTS seg {i} failed: {e}")
        finally:
            if os.path.exists(seg_path):
                os.remove(seg_path)

    # Final silence check
    max_val = np.abs(full_audio).max()
    if max_val < 0.001:
        raise RuntimeError(
            "TTS produced silent audio for ALL segments.\n"
            "Fix: pip install indic-transliteration\n"
            "Then restart the server."
        )

    # Normalize and save
    full_audio = full_audio / max_val * 0.95
    out_path   = os.path.join(output_dir, f"{job_id}_synthesized.wav")
    sf.write(out_path, full_audio, sample_rate)
    logger.info(f"[{job_id}] Audio saved: {out_path} ({total_dur:.1f}s)")
    return out_path