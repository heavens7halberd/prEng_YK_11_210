import sys
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
import io
import wave
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from main import app

client = TestClient(app)


TEST_TEXT_POSITIVE = "I love this movie! It's amazing!"
TEST_TEXT_NEGATIVE = "I hate rainy days, they make me depressed."
TEST_IMAGE_SIZE = (224, 224)

def generate_test_audio():
    """Генерация тестового WAV аудио в памяти"""
    sample_rate = 44100
    duration = 1.0  
    frequency = 440.0  
    
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.5
    
    
    audio_bytes = io.BytesIO()
    with wave.open(audio_bytes, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    
    audio_bytes.seek(0)
    return audio_bytes

def generate_test_video():
    """Генерация тестового MP4 видео в памяти (миниатюрный видеофайл)"""
    
    video_data = io.BytesIO()
    video_data.write(b'fake video data')  
    video_data.seek(0)
    return video_data

def test_get_models():
    """Тест для GET / (получение списка моделей)"""
    response = client.get("/")
    assert response.status_code == 200
    assert all(model in response.text for model in ["tone", "image", "audio", "video"])

def test_tone_analysis_positive():
    """Тест для POST /tone/ (анализ тональности, позитивный текст)"""
    response = client.post("/tone/", json={"text": TEST_TEXT_POSITIVE})
    assert response.status_code == 200
    result = response.json()
    assert result["label"] in ["POSITIVE", "NEGATIVE"]  
    assert isinstance(result["score"], float)

def test_tone_analysis_negative():
    """Тест для POST /tone/ (анализ тональности, негативный текст)"""
    response = client.post("/tone/", json={"text": TEST_TEXT_NEGATIVE})
    assert response.status_code == 200
    result = response.json()
    assert result["label"] in ["POSITIVE", "NEGATIVE"]
    assert isinstance(result["score"], float)

def test_image_classification():
    """Тест для POST /image/ (классификация изображения)"""
    img = Image.new("RGB", TEST_IMAGE_SIZE, color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    response = client.post(
        "/image/",
        files={"body": ("test_image.jpg", img_byte_arr, "image/jpeg")}
    )
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, str)

def test_audio_transcription():
    """Тест для POST /audio/ (транскрибация аудио)"""
    audio_data = generate_test_audio()
    
    response = client.post(
        "/audio/",
        files={"body": ("test_audio.wav", audio_data, "audio/wav")}
    )
    assert response.status_code == 200
    result = response.json()
    
    assert isinstance(result, (str, dict)) 

def test_video_classification():
    """Тест для POST /video/ (классификация видео)"""
    video_data = generate_test_video()
    
    response = client.post(
        "/video/",
        files={"body": ("test_video.mp4", video_data, "video/mp4")}
    )
    assert response.status_code == 200
    result = response.json()
    
    assert isinstance(result, (str, dict))

def test_invalid_content_types():
    """Тест обработки неверных типов контента"""
    
    response = client.post(
        "/image/",
        files={"body": ("test.txt", io.BytesIO(b"text"), "text/plain")}
    )
    assert response.status_code == 400
    
    
    response = client.post(
        "/audio/",
        files={"body": ("test.txt", io.BytesIO(b"text"), "text/plain")}
    )
    assert response.status_code == 400
    
    
    response = client.post(
        "/video/",
        files={"body": ("test.txt", io.BytesIO(b"text"), "text/plain")}
    )
    assert response.status_code == 400