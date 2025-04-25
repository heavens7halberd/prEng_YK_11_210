from typing import Annotated
import json

from fastapi import FastAPI, Body, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
import uvicorn
from pydantic import BaseModel, Field

from ml_models.tone import tone
from ml_models.image import image
from ml_models.audio import audio
from ml_models.video import video


app = FastAPI()


ml_models = {
    "tone": "Определение тональности текста",
    "image": "Определение объекта на изображении",
    "audio": "Преобразование аудио в текст",
    "video": "Классификация событий в видео"
}


class UserIn(BaseModel):
    text: str = Field(min_length=8)


@app.get("/")
def get_models() -> str:
    '''
    Return all available models, JSON formatted.
    '''
    response_obj = json.dumps(ml_models, ensure_ascii=False)
    return response_obj


@app.post("/tone/")
def tone_func(body: Annotated[UserIn, Body()]):
    '''
    Return result of tone.analyze_tone func from ml_models.
    '''
    try:
        analyzed_text = tone.analyze_tone(body.text)
        response_obj = jsonable_encoder(analyzed_text)
        return response_obj
    except Exception as e:
        return {"error": str(e)}
    

@app.post("/image/")
async def image_func(body: UploadFile):
    '''
    Return result of image.classify_image func from ml_models.
    '''
    if body.content_type == 'image/jpeg':
        try:
            content = await body.read()
            analyzed_image = image.classify_image(content)
            response_obj = jsonable_encoder(analyzed_image)
            return response_obj
        except Exception as e:
            return {"error": str(e)}
    raise HTTPException(status_code=400, detail="Invalid object type, expected 'image/jpeg'")


@app.post("/audio/")
async def audio_func(body: UploadFile):
    '''
    Return result of audio.transcribe_audio func from ml_models.
    '''
    if body.content_type == 'audio/wav':
        try:
            content = await body.read()
            analyzed_audio = audio.transcribe_audio(content)
            response_obj = jsonable_encoder(analyzed_audio)
            return response_obj
        except Exception as e:
            return {"error": str(e)}
    raise HTTPException(status_code=400, detail="Invalid object type, expected 'audio/wav'")
    

@app.post("/video/")
async def video_func(body: UploadFile):
    '''
    Return result of video.predict_video_class func from ml_models.
    '''
    if body.content_type == 'video/mp4':
        try:
            content = await body.read()
            analyzed_video = video.predict_video_class(content)
            response_obj = jsonable_encoder(analyzed_video)
            return response_obj
        except Exception as e:
            return {"error": str(e)}
    raise HTTPException(status_code=400, detail="Invalid object type, expected 'video/mp4'")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
