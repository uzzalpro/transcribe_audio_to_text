import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pytest
from unittest.mock import patch
from app import app
from utils.asr_local import transcribe_local

#############################-------------------##################

''' Unit test for the adapter ''' 

#############################-------------------##################

def test_transcribe_local_output_format(monkeypatch):
    # Mock the WhisperModel and its transcribe method
    class MockWhisperModel:
        def transcribe(self, filename, language=None, word_timestamps=True):
            class Segment:
                def __init__(self):
                    self.start = 0.0
                    self.end = 1.5
                    self.text = "Hello world."
                    self.words = [
                        type("Word", (), {"start": 0.0, "end": 0.5, "word": "Hello"})(),
                        type("Word", (), {"start": 0.6, "end": 1.5, "word": "world"})()
                    ]
            return [Segment()], None

    def mock_init(model_name, compute_type=None):
        return MockWhisperModel()

    # Patch WhisperModel constructor
    monkeypatch.setattr("utils.asr_local.WhisperModel", mock_init)

    # Pass dummy audio bytes (won't be used due to mock)
    dummy_audio = b"dummy audio bytes"
    result = transcribe_local(dummy_audio)

    assert "text" in result
    assert "segments" in result
    assert isinstance(result["segments"], list)
    segment = result["segments"][0]
    assert segment["start"] == 0.0
    assert segment["end"] == 1.5
    assert "words" in segment
    assert len(segment["words"]) == 2
    assert segment["words"][0]["word"] == "Hello"



#############################-------------------##################

''' Integration test using Flask test client ''' 

#############################-------------------##################

@pytest.fixture
def client():

    os.environ["WHISPER_PROVIDER"] = "faster-whisper"
    # app.config["TESTING"] = True
    return app.test_client()

def test_build_srt_file_route():
    dummy_audio = b"dummy audio bytes"

    with app.test_client() as client:
        with patch("app.get_whisper_srt") as mock_get_whisper_srt:
            mock_get_whisper_srt.return_value = {
                "text": "Mocked transcript",
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "text": "Mocked transcript",
                        "words": [{"start": 0.0, "end": 1.0, "word": "Mocked"}]
                    }
                ]
            }

            response = client.post(
                "/build_srt_file/1",
                data=dummy_audio,
                headers={"Content-Type": "audio/wav"},
            )

            assert response.status_code == 200
            json_data = response.get_json()
            assert json_data["data"]["text"] == "Mocked transcript"
            assert json_data["message"] == "SRT created"

