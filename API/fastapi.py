from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

app = FastAPI()

# Set up some models so we can pick what we want to use
def load_toxic_comment_model():
    # Load the 'toxic comment' model from Hugging Face
    model_path = "martin-ha/toxic-comment-model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return TextClassificationPipeline(model=model, tokenizer=tokenizer)

def load_BERT_model():
    # Load the BERT model
    model_path = 'nlptown/bert-base-multilingual-uncased-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return TextClassificationPipeline(model=model, tokenizer=tokenizer)

pipeline = load_BERT_model()

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

    # Get sentiment
    sentiment = pipeline(message)[0]
    # There will be some predict code here in future but for now just return a random number
    return {"comment": message, "sentiment": sentiment['label'], "confidence": sentiment['score']}
