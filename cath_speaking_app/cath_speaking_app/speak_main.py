import os
import io
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from ichigo.asr import transcribe
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# --- In-Memory Speech-to-Text Utility (with temp file) ---
def memory_speech_to_text_from_array(audio_array, sample_rate) -> str:
    """
    Nhận numpy audio array và sample_rate,
    ghi ra WAV tạm file và chạy transcribe trên file.
    """
    # Xuất WAV vào buffer
    buf = io.BytesIO()
    wav_write(buf, sample_rate, audio_array)
    buf.seek(0)

    # Ghi buffer xuống file tạm
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(buf.read())
        tmp_path = tmp.name

    try:
        result = transcribe(tmp_path)
        transcript = result[0] if isinstance(result, tuple) and result else str(result)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return transcript

# --- Audio Recording Utility ---
RECORD_DURATION = 5  # seconds
SAMPLE_RATE = 16000   # Hz

def record_audio(duration: int = RECORD_DURATION, sample_rate: int = SAMPLE_RATE):
    """Ghi âm trực tiếp từ mic, trả về numpy array và sample_rate."""
    print(f"🎤 Đang ghi âm {duration}s...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("✅ Ghi âm hoàn tất.")
    return recording.flatten(), sample_rate

# --- IELTS Speaking Trainer CLI with Voice In-Memory ---
class IELTSSpeakingTrainerCLI:
    def __init__(self, api_key: str):
        """Khởi tạo LLM và bộ nhớ hội thoại."""
        self.conv = ConversationChain(
            llm=ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=api_key,
                temperature=0.7
            ),
            memory=ConversationBufferMemory(),
            verbose=False
        )
        self.question_count = 0

    def display_welcome(self):
        print("\n===== IELTS SPEAKING TRAINER (CLI) =====")
        print("Bạn có thể trả lời bằng giọng nói.")
        print("Nhập 'r' để ghi âm 5s, hoặc gõ trực tiếp câu trả lời.")
        print("Dùng 'next' để chuyển phần, 'exit' để thoát.")

    def select_topic(self) -> str:
        topics = [
            "Education","Technology","Family","Environment",
            "Travel","Health","Work","Culture","Food","Sports"
        ]
        for i, t in enumerate(topics, 1):
            print(f"{i}. {t}")
        choice = input("Chọn chủ đề (số): ")
        return topics[int(choice)-1] if choice.isdigit() and 1 <= int(choice) <= len(topics) else None

    def get_part_prompt(self, part: int, topic: str) -> str:
        if part == 1:
            return f"You are an IELTS examiner for Part 1 on topic: {topic}. Ask simple, personal questions."  
        if part == 2:
            return f"Part 2: Describe a time when {topic.lower()} was important in your life."  
        return f"Part 3: Discuss broader analytical questions about {topic} in society."  

    def conduct(self, topic: str):
        for part in [1, 2, 3]:
            print(f"--- PART {part} ---")
            question = self.conv.predict(input=self.get_part_prompt(part, topic))
            print(f"Examiner: {question}\n")
            while True:
                cmd = input("Nhập 'r' để ghi âm, gõ text, hoặc 'next'/'exit': ")
                if cmd.lower() == 'exit':
                    print("Kết thúc luyện tập.")
                    return
                if cmd.lower() == 'next':
                    break
                if cmd.lower() == 'r':
                    audio_array, sr = record_audio()
                    user_text = memory_speech_to_text_from_array(audio_array, sr)
                else:
                    user_text = cmd
                response = self.conv.predict(input=user_text)
                print(f"Examiner: {response}\n")
                if part == 1:
                    self.question_count += 1
                    if self.question_count >= 5:
                        print("Hoàn thành Part 1. Gõ 'next' để sang Part 2.")

    def run(self):
        self.display_welcome()
        topic = self.select_topic()
        if not topic:
            print("Chủ đề không hợp lệ. Kết thúc.")
            return
        print(f"Chủ đề: {topic}")
        self.conduct(topic)

# --- Main Execution ---
if __name__ == '__main__':
    # Sử dụng API key trực tiếp
    API_KEY = "AIzaSyB3x4ETUv3x5LByfQKRV4P2ta6BqWO9TM0"
    IELTSSpeakingTrainerCLI(API_KEY).run()
