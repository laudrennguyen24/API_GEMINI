import os
import tempfile
from flask import Flask, request, jsonify, render_template
from pydub import AudioSegment
from ichigo.asr import transcribe
from speak_main import IELTSSpeakingTrainerCLI

app = Flask(__name__)

# Khởi trainer chỉ một lần
trainer = IELTSSpeakingTrainerCLI(
    api_key="AIzaSyB3x4ETUv3x5LByfQKRV4P2ta6BqWO9TM0"
)

@app.route("/")
def index():
    return render_template("web.html")

@app.route("/topics/")
def topics():
    return jsonify({
      "topics": [
        "Education","Technology","Family","Environment",
        "Travel","Health","Work","Culture","Food","Sports"
      ]
    })

@app.route("/start/", methods=["POST"])
def start():
    data = request.get_json()
    topic = data.get("topic", "")
    # reset memory & counter
    trainer.conv.memory.clear()
    trainer.question_count = 0

    prompt = trainer.get_part_prompt(1, topic)
    first_q = trainer.conv.predict(input=prompt)
    return jsonify({"question": first_q})

@app.route("/answer/", methods=["POST"])
def answer():
    data = request.get_json()
    user_text = data.get("text", "").strip()
    topic     = data.get("topic", "").strip()
    if not user_text:
        return jsonify({"error": "Empty text"}), 400

    next_q = trainer.conv.predict(input=user_text)

    # sau 5 câu Part1 → chuyển Part2
    if trainer.question_count < 5:
        trainer.question_count += 1
        if trainer.question_count == 5:
            next_q = trainer.conv.predict(
                input=trainer.get_part_prompt(2, topic)
            )
    return jsonify({"question": next_q})

@app.route("/upload-audio/", methods=["POST"])
def upload_audio():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    audio = AudioSegment.from_file(f)
    audio = audio.set_frame_rate(16000).set_channels(1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio.export(tmp.name, format="wav")
        tmp_path = tmp.name

    try:
        result = transcribe(tmp_path)
        transcript = result[0] if isinstance(result, tuple) else str(result)
    finally:
        os.remove(tmp_path)

    return jsonify({"transcript": transcript})

if __name__ == "__main__":
    # Truy cập http://localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
