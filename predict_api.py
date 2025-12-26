from fastapi import FastAPI, UploadFile
from model import load_model, predict_image_class
from utils import get_calories_for_food

app = FastAPI()

model = load_model()

@app.post("/predict")
async def predict(file: UploadFile):
    img_bytes = await file.read()

    food = predict_image_class(model, img_bytes)
    calories, source = get_calories_for_food(food)

    return {
        "food": food,
        "calories": calories,
        "source": source
    }
