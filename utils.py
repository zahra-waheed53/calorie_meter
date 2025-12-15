import os
import requests
import pandas as pd
import re

USDA_API_KEY = os.getenv("USDA_API_KEY")

FOUNDATION_DIR = "usda_dataset"   # your folder containing the CSV files

# ---------------------- LOAD FOUNDATION DATASET -----------------------------

def load_foundation_dataset():
    """
    Loads USDA Foundation dataset using the real file structure shown in screenshot.
    Merges:
        - foundation_food.csv
        - food_nutrient.csv
        - nutrient.csv
    into a table: description | kcal
    """

    # File paths based on your screenshot
    foundation_food_path = os.path.join(FOUNDATION_DIR, "foundation_food.csv")
    food_nutrient_path = os.path.join(FOUNDATION_DIR, "food_nutrient.csv")
    nutrient_path = os.path.join(FOUNDATION_DIR, "nutrient.csv")

    # Load CSVs
    foundation_food_df = pd.read_csv(foundation_food_path)
    food_nutrient_df = pd.read_csv(food_nutrient_path)
    nutrient_df = pd.read_csv(nutrient_path)

    # Identify the nutrient ID for calories (Energy, kcal)
    energy_ids = nutrient_df[
        (nutrient_df["name"].str.lower().str.contains("energy")) &
        (nutrient_df["unit_name"].str.lower().str.contains("kcal"))
    ]["id"].tolist()

    # Filter food_nutrient rows to those nutrients
    kcal_nutrients = food_nutrient_df[
        food_nutrient_df["nutrient_id"].isin(energy_ids)
    ][["fdc_id", "amount"]]

    # Merge kcal into foundation_food table
    merged = foundation_food_df.merge(
        kcal_nutrients,
        on="fdc_id",
        how="left"
    )

    merged = merged.rename(columns={"amount": "kcal"})

    return merged


# Load dataset once at startup
foundation_df = load_foundation_dataset()


# ---------------------- NORMALIZATION -------------------------------------

def normalize_food_name(food: str) -> str:
    food = food.lower()
    food = food.replace("_", " ").strip()
    food = re.sub(r"[^a-zA-Z0-9 ]+", "", food)
    return food


# ---------------------- USDA API LOOKUP -----------------------------------

def query_usda(food_name: str):
    try:
        url = "https://api.nal.usda.gov/fdc/v1/foods/search"

        params = {
            "api_key": USDA_API_KEY,
            "query": food_name,
            "dataType": ["Foundation"]
        }

        res = requests.get(url, params=params)
        if res.status_code != 200:
            return None

        data = res.json()
        if "foods" not in data or len(data["foods"]) == 0:
            return None

        for nutrient in data["foods"][0].get("foodNutrients", []):
            if nutrient.get("nutrientName") == "Energy" and nutrient.get("unitName") == "KCAL":
                return nutrient.get("value")

        return None
    except:
        return None


# ---------------------- LOCAL LOOKUP (FOUNDATION DATASET) ------------------

def search_local_foundation_csv(food_name: str):
    matches = foundation_df[
        foundation_df["description"].str.lower().str.contains(food_name)
    ]

    if matches.empty:
        return None

    row = matches.iloc[0]
    return row.get("kcal", None)


# ---------------------- MAIN LOOKUP PIPELINE ------------------------------

def get_calories_for_food(predicted_class: str):
    food_name = normalize_food_name(predicted_class)

    # 1. USDA API
    calories = query_usda(food_name)
    if calories is not None:
        return calories, "USDA API"

    # 2. Local USDA Foundation Dataset (merged CSV)
    calories = search_local_foundation_csv(food_name)
    if calories is not None:
        return calories, "USDA Foundation Dataset"

    return None, "Not Found"
