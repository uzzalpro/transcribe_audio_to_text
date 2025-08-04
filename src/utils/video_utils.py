# import os
# from .asr_local import transcribe_local
# # from .openai_asr import transcribe_openai  # assume existing OpenAI logic here

# def get_whisper_srt(f_name: str, audio_bytes: bytes) -> dict:
#     provider = os.getenv("WHISPER_PROVIDER", "openai")

#     if provider == "openai":
#         return transcribe_openai(audio_bytes)
#     elif provider == "faster-whisper":
#         return transcribe_local(audio_bytes)
#     elif provider == "auto":
#         try:
#             return transcribe_openai(audio_bytes)
#         except Exception as e:
#             print(f"[Fallback to local ASR] OpenAI failed: {e}")
#             return transcribe_local(audio_bytes)
#     else:
#         raise ValueError("Invalid WHISPER_PROVIDER value.")

import os
from .asr_local import transcribe_local

def transcribe_openai(audio_bytes):
    raise RuntimeError("OpenAI not available in local-only mode")

def get_whisper_srt(f_name: str, audio_bytes: bytes) -> dict:
    provider = os.getenv("WHISPER_PROVIDER", "openai")

    if provider == "openai":
        return transcribe_openai(audio_bytes)
    elif provider == "faster-whisper":
        return transcribe_local(audio_bytes)
    elif provider == "auto":
        try:
            return transcribe_openai(audio_bytes)
        except Exception as e:
            print(f"[Fallback to local ASR] OpenAI failed: {e}")
            return transcribe_local(audio_bytes)
    else:
        raise ValueError("Invalid WHISPER_PROVIDER value.")
