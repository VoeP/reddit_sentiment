from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

app = FastAPI()

CHUNK_SIZE = 512

# Generic error function to use for all endpoints
def message_error(message):
    if message is None:
        return {"error": "No comment provided"}
    return None

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

pipeline_bert = load_BERT_model()
pipeline_toxic = load_toxic_comment_model()

# Create the root endpoint
@app.get("/")
def index():
    return {"message": "Hello, World!"}

# Create the predict endpoint
# Example usage in browser: http://localhost:8000/predict?message=Hello
@app.get("/predict_bert")
def predict_bert(message):
    error = message_error(message)
    if error is not None:
        return error

    # Split message into chunks
    sentiments = []
    confidences = []
    for i in range(0, len(message), CHUNK_SIZE):
        chunk = message[i:i+CHUNK_SIZE]
        # Get sentiment
        sentiment = pipeline_bert(chunk)[0]
        # Add to the list
        sentiments.append(sentiment['label'])
        confidences.append(sentiment['score'])

    print("Processed chunks: ", len(sentiments))

    # Return the average sentiment and confidence
    # Sentiment is of a certain form so we need to modify it a little
    # For the Bert model it is "n stars" so take the first char and convert to int
    avg_sentiment = round(sum([int(s[0]) for s in sentiments]) / len(sentiments))
    # Convert back to the "n stars" format
    sentiment = str(avg_sentiment) + " stars"
    avg_confidence = sum(confidences) / len(confidences)

    # There will be some predict code here in future but for now just return a random number
    return {"comment": message, "sentiment": sentiment, "confidence": avg_confidence}

@app.get("/predict_toxic")
def predict_toxic(message):
    error = message_error(message)
    if error is not None:
        return error

    # chunk message
    sentiments = []
    confidences = []
    for i in range(0, len(message), CHUNK_SIZE):
        chunk = message[i:i+CHUNK_SIZE]
        # Get sentiment
        sentiment = pipeline_toxic(chunk)[0]
        # Add to the list
        sentiments.append(sentiment['label'])
        confidences.append(sentiment['score'])

    print("Processed chunks: ", len(sentiments))

    # Sentiment is the most frequently predicted label
    # This WON'T account for a tie but we probably won't use this model so that's fine
    sentiment = max(set(sentiments), key = sentiments.count)
    avg_confidence = sum(confidences) / len(confidences)

    return {"comment": message, "sentiment": sentiment, "confidence": avg_confidence}
