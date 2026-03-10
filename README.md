🎬 Auto Dubbing for Educational Videos
AI-Powered Multilingual Video Dubbing (Konkani • Maithili • Hindi)

An end-to-end AI system that automatically translates and dubs educational videos into regional Indian languages using speech recognition, machine translation, voice synthesis, and lip synchronization.

🚀 Project Overview

This project builds a fully automated dubbing pipeline that:

🎙 Extracts speech from an input video

📝 Transcribes speech using Whisper

🌍 Translates into target language (Konkani / Maithili / Hindi)

🗣 Generates realistic speech using AI Text-to-Speech

👄 Synchronizes lips with generated audio using Wav2Lip

🎥 Outputs a fully dubbed and lip-synced video

This enables educational content to be accessible in regional languages.

🧠 Technologies Used

🎙 OpenAI Whisper – Speech-to-Text transcription

🌍 mT5 (Multilingual T5) – Language translation

🔊 Coqui TTS – Text-to-Speech voice generation

👄 Wav2Lip – Lip synchronization

🎬 FFmpeg – Video & audio processing

🐍 Python

🏗 System Architecture
Input Video
     ↓
Audio Extraction (FFmpeg)
     ↓
Speech-to-Text (Whisper)
     ↓
Translation (mT5)
     ↓
Text-to-Speech (Coqui TTS)
     ↓
Lip Sync (Wav2Lip)
     ↓
Final Dubbed Video
📂 Project Structure
Auto-Dubbing/
│
├── input_videos/
├── extracted_audio/
├── transcripts/
├── translated_text/
├── generated_audio/
├── lip_synced_output/
├── models/
├── main.py
└── README.md
⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/your-username/auto-dubbing.git
cd auto-dubbing
2️⃣ Install Dependencies
pip install -r requirements.txt

Install FFmpeg:

sudo apt install ffmpeg
▶️ How to Run
Step 1: Provide Input Video

Place your video inside:

input_videos/
Step 2: Run Pipeline
python main.py --video input.mp4 --language hindi

Supported Languages:

hindi

konkani

maithili

📌 Example Use Case

Input: English educational lecture
Output: Same lecture dubbed into Hindi with lip-sync

🎯 Features

✅ Fully automated pipeline
✅ Supports multiple Indian languages
✅ Works on educational content
✅ End-to-end AI integration
✅ Lip-synced output

📊 Sample Output

Original Video → English Lecture

Dubbed Version → Hindi synchronized output

(Add screenshots or video demo link here if available)

🧩 Challenges Faced

Accurate translation for low-resource languages

Lip-sync alignment issues

Managing voice naturalness

Synchronizing speech timing

🔮 Future Improvements

Add more Indian regional languages

Real-time dubbing

Speaker voice cloning

Web-based UI interface

Subtitle generation

🎓 Academic Relevance

This project demonstrates practical implementation of:

NLP (Speech Recognition & Translation)

Deep Learning

Generative AI

Computer Vision

Multimedia Processing

📜 License

This project is developed for academic and research purposes.

👨‍💻 Author

Ayyankula Lalith Sai Kumar
B.Tech CSE (AIML)
KIET Group of Institutions
