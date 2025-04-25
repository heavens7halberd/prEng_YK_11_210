from typing import Annotated
import json

from fastapi import FastAPI, Body, status, UploadFile, File
from fastapi.encoders import jsonable_encoder
import uvicorn
from pydantic import BaseModel, Field

from ml_models.tone import tone
from ml_models.image import image


app = FastAPI()


ml_models = {
    "tone": "Определение тональности текста",
    "image": "Определение объекта на изображении"
}


class UserIn(BaseModel):
    text: str = Field(min_length=8)


@app.get("/", status_code=status.HTTP_200_OK)
def get_models() -> str:
    response_obj = json.dumps(ml_models, ensure_ascii=False)
    return response_obj


@app.post("/tone/", status_code=status.HTTP_200_OK)
def tone_func(body: Annotated[UserIn, Body()]):
    try:
        analyzed_text = tone.analyze_tone(body.text)
        response_obj = jsonable_encoder(analyzed_text)
        return response_obj
    except Exception as e:
        return {"error": str(e)}
    

@app.post("/image/", status_code=status.HTTP_200_OK)
async def image_func(body: UploadFile):
    try:
        content = await body.read()
        analyzed_image = image.classify_image(content)
        response_obj = jsonable_encoder(analyzed_image)
        return response_obj
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
