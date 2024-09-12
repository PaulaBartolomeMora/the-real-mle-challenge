from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel
from typing import Literal
import pickle
import numpy as np
import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

DIR_REPO = Path.cwd().parent.parent
DIR_MODELS = Path(DIR_REPO) / "models"
MODEL_PATH = DIR_MODELS / "simple_classifier.pkl"


with open(MODEL_PATH, 'rb') as model_file:
    clf = pickle.load(model_file)


app = FastAPI()


class Listing(BaseModel):
    id: int
    accommodates: int
    room_type: Literal['Shared room', 'Private room', 'Entire home/apt', 'Hotel room']
    beds: int
    bedrooms: int
    bathrooms: float
    neighbourhood: Literal['Bronx', 'Queens', 'Staten Island', 'Brooklyn', 'Manhattan']
    tv: int
    elevator: int
    internet: int
    latitude: float
    longitude: float


MAP_ROOM_TYPE = {"Shared room": 1, "Private room": 2, "Entire home/apt": 3, "Hotel room": 4}
MAP_NEIGHB = {"Bronx": 1, "Queens": 2, "Staten Island": 3, "Brooklyn": 4, "Manhattan": 5}
PRICE_CATEGORY_MAP = {0: "Low", 1: "Mid", 2: "High", 3: "Lux"}


@app.get("/")
def home():
    logging.info("Accessed home page")
    return "Welcome to the price prediction API"


# PREDICCIÓN DE LA CATEORÍA DE PRECIOS
@app.post("/predict")
async def predict_price_category(listing: Listing, request: Request):
    logging.info(f"Prediction request received: {listing.dict()} from {request.client.host}")
    
    try:
        input_data = [
            MAP_NEIGHB[listing.neighbourhood],
            MAP_ROOM_TYPE[listing.room_type],
            listing.accommodates,
            listing.bathrooms,
            listing.bedrooms
        ]

        logging.info(f"Input data transformed: {input_data}")
        
        input_data = np.array(input_data).reshape(1, -1)

        predicted_class = clf.predict(input_data)[0]
        price_category = PRICE_CATEGORY_MAP[predicted_class]

        logging.info(f"Prediction made: {price_category} for ID: {listing.id}")

        return {
            "id": listing.id,
            "price_category": price_category
        }
    
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return {"error": "An error occurred while processing the prediction"}


# PARA INICIALIZAR DESDE EL CÓDIGO
""" if __name__ == "__main__":
    logging.info("Starting prediction API")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") """
