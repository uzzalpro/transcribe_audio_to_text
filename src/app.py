import os
from flask import Flask, request, jsonify
from utils.video_utils import get_whisper_srt

def load_env_vars():
    from dotenv import load_dotenv
    load_dotenv(".env.dev")  # or switch to "test.env" for testing
    os.environ.setdefault("WHISPER_PROVIDER", "auto")

# Load environment variables before app is created
load_env_vars()

app = Flask(__name__)

@app.route("/build_srt_file/<int:project_id>", methods=["POST"])
def build_srt_file(project_id):
    audio_bytes = request.data
    try:
        transcription = get_whisper_srt(f"project_{project_id}.wav", audio_bytes)
        return jsonify({"message": "SRT created", "data": transcription}), 200
    except Exception as e:
        import traceback
        print("ERROR:", e)
        traceback.print_exc()  # 👈 this will show full error
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
