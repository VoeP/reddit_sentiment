from fastapi import FastAPI
from random import random

app = FastAPI()

# Create the root endpoint
@app.get("/")
def index():
    return {"message": "Hello, World!"}

# Create the predict endpoint
# Example usage in browser: http://localhost:8000/predict?message=Hello
@app.get("/predict")
def predict(message):
    if message is None:
        return {"error": "No comment provided"}

    # There will be some predict code here in future but for now just return a random number
    return {"comment": message, "sentiment": random() * 2 - 1}
