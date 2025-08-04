# transcribe_audio_to_text
## ASR Provider Switch

You can choose between OpenAI Whisper and local transcription using faster-whisper:

### 🧠 OpenAI Whisper
- Set in `dev.env`:
  ```env
  WHISPER_PROVIDER=openai
  OPENAI_API_KEY=your_key


- Test by CURL

```
curl -X POST http://127.0.0.1:5000/build_srt_file/1 --data-binary @sample.wav -H "Content-Type: audio/wav"

```