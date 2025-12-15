from fastapi import FastAPI, UploadFile, Form, HTTPException
from model import load_model, predict_image_class
from utils import get_calories_for_food

app = FastAPI()
model = load_model()

# Average serving sizes (can be justified in viva)
SERVING_GRAMS = 100

# Preset portion sizes (grams)
PORTION_PRESETS = {
    "small": 100,
    "medium": 200,
    "large": 300
}

@app.post("/predict")
async def predict(
    file: UploadFile,

    # quantity inputs
    grams: float = Form(None),
    servings: float = Form(None),
    preset: str = Form(None)  # small / medium / large
):
    img_bytes = await file.read()

    # ---------- Food Prediction ----------
    food = predict_image_class(model, img_bytes)
    kcal_100g, source = get_calories_for_food(food)

    if kcal_100g is None:
        raise HTTPException(status_code=404, detail="Calories not found")

    # ---------- Portion Resolution ----------
    if preset:
        preset = preset.lower()
        if preset not in PORTION_PRESETS:
            raise HTTPException(
                status_code=400,
                detail="Preset must be one of: small, medium, large"
            )
        portion_grams = PORTION_PRESETS[preset]

    elif servings is not None:
        if servings <= 0:
            raise HTTPException(
                status_code=400,
                detail="Servings must be greater than 0"
            )
        portion_grams = servings * SERVING_GRAMS

    else:
        # default to grams
        if grams is None:
            grams = 100  # safe default

        if grams <= 0:
            raise HTTPException(
                status_code=400,
                detail="Grams must be greater than 0"
            )

        portion_grams = grams

    # ---------- Calorie Calculation ----------
    total_calories = (kcal_100g / 100) * portion_grams

    return {
        "food": food,
        "calories_per_100g": kcal_100g,
        "portion": {
            "grams": portion_grams,
            "source": (
                "preset" if preset else
                "servings" if servings is not None else
                "grams"
            )
        },
        "total_calories": round(total_calories, 2),
        "calorie_source": source
    }
