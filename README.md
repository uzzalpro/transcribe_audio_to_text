# transcribe_audio_to_text
## ASR Provider Switch

You can choose between OpenAI Whisper and local transcription using faster-whisper:

### 🧠 OpenAI Whisper
- Set in `dev.env`:
  ```env
  WHISPER_PROVIDER=openai
  OPENAI_API_KEY=your_key
