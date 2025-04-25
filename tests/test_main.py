import sys
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
import io


sys.path.append(str(Path(__file__).parent.parent))

from main import app

client = TestClient(app)

TEST_TEXT_POSITIVE = "I love this movie! It's amazing!"
TEST_TEXT_NEGATIVE = "I hate rainy days, they make me depressed."
TEST_IMAGE_SIZE = (224, 224)

def test_get_models():
    """Тест для GET / (получение списка моделей)"""

    response = client.get("/")
    assert response.status_code == 200
    assert "tone" in response.text
    assert "image" in response.text

def test_tone_analysis_positive():
    """Тест для POST /tone/ (анализ тональности, позитивный текст)"""

    response = client.post("/tone/", json={"text": TEST_TEXT_POSITIVE})
    assert response.status_code == 200
    result = response.json()
    assert result["label"] == "POSITIVE"
    assert isinstance(result["score"], float)

def test_tone_analysis_negative():
    """Тест для POST /tone/ (анализ тональности, негативный текст)"""

    response = client.post("/tone/", json={"text": TEST_TEXT_NEGATIVE})
    assert response.status_code == 200
    result = response.json()
    assert result["label"] == "NEGATIVE"
    assert isinstance(result["score"], float)

def test_tone_analysis_invalid_input():
    """Тест для POST /tone/ (некорректный ввод)"""

    response = client.post("/tone/", json={"text": "short"})
    assert response.status_code == 422

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

def test_image_classification_invalid_input():
    """Тест для POST /image/ (некорректный ввод)"""

    response = client.post(
        "/image/",
        files={"body": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    )
    assert response.status_code == 400
