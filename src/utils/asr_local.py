from faster_whisper import WhisperModel

def transcribe_local(audio_bytes: bytes, language: str = None) -> dict:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        model = WhisperModel("tiny", compute_type="int8")  # or "base"
        segments, info = model.transcribe(tmp.name, language=language, word_timestamps=True)

        results = {"text": "", "segments": []}
        for i, seg in enumerate(segments):
            words = [{"start": w.start, "end": w.end, "word": w.word} for w in seg.words]
            results["segments"].append({
                "id": i,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": words
            })
            results["text"] += seg.text + " "
        return results
