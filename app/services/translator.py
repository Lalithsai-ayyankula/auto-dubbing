"""
Translation Service
Step 3: Translate Hindi text to Konkani or Maithili using mT5

Model strategy:
- We use Helsinki-NLP/opus-mt-hi-* models which are practical multilingual models.
- For Konkani (gom) and Maithili (mai), we use mT5-base with language prompting
  as a multilingual fallback, since direct Hindi→Konkani/Maithili fine-tuned models
  are rare in the open-source ecosystem.
- If a fine-tuned checkpoint exists at models/mt5-konkani or models/mt5-maithili,
  it will be loaded automatically (drop your fine-tuned checkpoint there).
"""

import logging
import os
import torch
from transformers import MT5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)

# Language code mapping for mT5 prompting strategy
LANGUAGE_CONFIG = {
    "konkani": {
        "lang_code": "gom",
        "prompt_prefix": "translate Hindi to Konkani: ",
        "local_model_dir": "models/mt5-konkani",
        "hf_model_id": "google/mt5-base",  # Fallback if no fine-tuned model
    },
    "maithili": {
        "lang_code": "mai",
        "prompt_prefix": "translate Hindi to Maithili: ",
        "local_model_dir": "models/mt5-maithili",
        "hf_model_id": "google/mt5-base",  # Fallback if no fine-tuned model
    },
}

# Cache per language
_translation_models: dict = {}
_translation_tokenizers: dict = {}


def get_translation_model(language: str):
    """
    Load or retrieve cached translation model and tokenizer.
    Prefers locally fine-tuned checkpoint; falls back to google/mt5-base.
    """
    if language not in _translation_models:
        config = LANGUAGE_CONFIG[language]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check for local fine-tuned checkpoint
        if os.path.isdir(config["local_model_dir"]):
            model_id = config["local_model_dir"]
            logger.info(f"Loading fine-tuned model from: {model_id}")
        else:
            model_id = config["hf_model_id"]
            logger.info(
                f"No fine-tuned model found at '{config['local_model_dir']}'. "
                f"Loading base model: {model_id}"
            )

        tokenizer = T5Tokenizer.from_pretrained(model_id)
        model = MT5ForConditionalGeneration.from_pretrained(model_id)
        model.to(device)
        model.eval()

        _translation_models[language] = (model, device)
        _translation_tokenizers[language] = tokenizer
        logger.info(f"Translation model for '{language}' loaded on {device}")

    model, device = _translation_models[language]
    tokenizer = _translation_tokenizers[language]
    return model, tokenizer, device


def translate_segments(segments: list[dict], language: str, job_id: str) -> list[dict]:
    """
    Translate a list of transcription segments from Hindi to target language.

    Args:
        segments: List of {'start', 'end', 'text'} dicts (Hindi text)
        language: 'konkani' or 'maithili'
        job_id: For logging

    Returns:
        Same structure with 'text' replaced by translated text,
        and 'original_text' added for reference.
    """
    logger.info(f"[{job_id}] Translating {len(segments)} segments to {language}")
    model, tokenizer, device = get_translation_model(language)
    config = LANGUAGE_CONFIG[language]
    prefix = config["prompt_prefix"]

    translated_segments = []

    for i, seg in enumerate(segments):
        hindi_text = seg["text"].strip()
        if not hindi_text:
            translated_segments.append({**seg, "original_text": hindi_text})
            continue

        # Prepend language prompt for mT5 conditional generation
        input_text = prefix + hindi_text

        try:
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=4,             # Beam search for quality
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

            translated_text = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            ).strip()

            logger.debug(f"[{job_id}] Seg {i}: '{hindi_text}' → '{translated_text}'")

        except Exception as e:
            logger.warning(f"[{job_id}] Translation failed for segment {i}: {e}. Using original.")
            translated_text = hindi_text  # Graceful fallback

        translated_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": translated_text,
            "original_text": hindi_text,
        })

    logger.info(f"[{job_id}] Translation complete for {len(translated_segments)} segments")
    return translated_segments
