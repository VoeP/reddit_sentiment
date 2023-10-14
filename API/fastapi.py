from fastapi import FastAPI
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import pandas as pd
from google.cloud import bigquery
from datetime import datetime

# Create the app object
app = FastAPI()

# Define constants
CHUNK_SIZE = 512
CACHE_UPDATE = datetime.now()

# Generic error function to use for all endpoints
def message_error(message):
    if message is None:
        return {"error": "No comment provided"}
    return None

# Generic function for loading a huggingface model
def load_huggingface_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="/app/cache")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir="/app/cache")

    return TextClassificationPipeline(model=model, tokenizer=tokenizer)

# Generic function for getting tokenizer and model from huggingface
def load_huggingface_tokenizer_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="/app/cache")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir="/app/cache")

    return tokenizer, model


pipeline_bert = load_huggingface_model("nlptown/bert-base-multilingual-uncased-sentiment")
emotions_tokeniser, emotions_model = load_huggingface_tokenizer_model("cardiffnlp/twitter-roberta-base-emotion")


def get_wsb_data(override_cache=False):
    # Check if we have a cached version of the data
    # It's unlikely an instance will be running for more than an hour
    # But also since this gets called 3 times by the different endpoints we might as well cache it
    # Especially as the streamlit site will be calling all 3 endpoints
    cache_expiry = 3600 # 1 hour
    if not override_cache:
        dt = datetime.now() - CACHE_UPDATE
        if dt.seconds < cache_expiry:
            print("Using cached data")
            return comments_df, posts_df

    # Constants
    project_id = "reddit-sentiment-400608"
    dataset_id = "wallstreetbets"
    table_id = "reddit_comments"
    # Bigquery client
    client = bigquery.Client(project=project_id)
    print("Bigquery connection established")

    # Set up the query
    query = f"""
    SELECT *
    FROM {project_id}.{dataset_id}.{table_id}
    WHERE date = MAX(date)"""

    # Run the query
    query_job = client.query(query)
    comment_df = query_job.to_dataframe()

    # Switch table id to posts
    table_id = "reddit_posts"
    # New query
    query = f"""
    SELECT *
    FROM {project_id}.{dataset_id}.{table_id}
    WHERE date = MAX(date)"""

    # Run the query
    query_job = client.query(query)
    post_df = query_job.to_dataframe()

    # Done
    return comment_df, post_df


# Let's do an initial query to get the latest data
comments_df, posts_df = get_wsb_data(True)


# Create the root endpoint
@app.get("/")
def index():
    return {"message": "Hello, World!"}

# Create the predict endpoint
# Example usage in browser: http://localhost:8000/predict?message=Hello
@app.get("/predict_message")
def predict_message(message):
    error = message_error(message)
    if error is not None:
        return error

    # Split message into chunks
    sentiments = []
    confidences = []
    emotions = []
    for i in range(0, len(message), CHUNK_SIZE):
        chunk = message[i:i+CHUNK_SIZE]
        # Get sentiment
        sentiment = pipeline_bert(chunk)[0]
        # Add to the list
        sentiments.append(sentiment['label'])
        confidences.append(sentiment['score'])

        # Get the emotions - they will just be a dictionary
        inputs = emotions_tokeniser(message, return_tensors="pt")
        with torch.no_grad():
            logits = emotions_model(**inputs).logits.tolist()[0]
            emotions.append({"joy": logits[0], "optimism": logits[1], "anger": logits[2], "sadness": logits[3]})


    print("Processed chunks: ", len(sentiments))

    # Return the average sentiment and confidence
    # Sentiment is of a certain form so we need to modify it a little
    # For the Bert model it is "n stars" so take the first char and convert to int
    avg_sentiment = round(sum([int(s[0]) for s in sentiments]) / len(sentiments))
    # Convert back to the "n stars" format
    sentiment = str(avg_sentiment) + " stars"
    avg_confidence = sum(confidences) / len(confidences)

    # Emotions are just averaged over each emotion in the list
    avg_emotions = {}
    for emotion in ["joy", "optimism", "anger", "sadness"]:
        avg_emotions[emotion] = sum([e[emotion] for e in emotions]) / len(emotions)

    # There will be some predict code here in future but for now just return a random number
    return {"comment": message, "sentiment": sentiment, "sentiment_confidence": avg_confidence, "emotions": avg_emotions}


@app.get("/wsb_emotions")
def wsb_sentiment():
    comment_df, post_df = get_wsb_data()

    # Create a dictionary to return
    return_dict = {}
    # Add the emotions
    return_dict['joy'] = comment_df['joy'].sum()
    return_dict['optimism'] = comment_df['optimism'].sum()
    return_dict['anger'] = comment_df['anger'].sum()
    return_dict['sadness'] = comment_df['sadness'].sum()

    # Return it
    return return_dict

@app.get("/wsb_emotions_by_post")
def wsb_emotions_by_post():
    comment_df, post_df = get_wsb_data()

    # Merge the dataframes on comment_df['post'], post_df['titles'] to include ID
    comment_df = comment_df.merge(post_df[['ids', 'titles']], left_on='post', right_on='titles')

    # Group the dataframe by post id and aggregate the emotions
    grouped_df = comment_df.groupby('ids').agg({'sentiment': 'mean', 'joy': 'sum', 'optimism': 'sum', 'anger': 'sum', 'sadness': 'sum', 'post': 'first', 'url': 'first'})

    # Convert emotion columns such that sum of all emotions is 1
    # Iterate over all rows to do this (slow af but it works)
    for _, row in grouped_df.iterrows():
        # Get the sum of all emotions
        sum_emotions = row['joy'] + row['optimism'] + row['anger'] + row['sadness']
        # Divide each emotion by the sum
        row['joy'] /= sum_emotions
        row['optimism'] /= sum_emotions
        row['anger'] /= sum_emotions
        row['sadness'] /= sum_emotions

    # Convert to dictionary and return
    return grouped_df.to_dict(orient='index')

@app.get("/wsb_sentiment_barplots_data")
def wsb_sentiment_barplots_data():
    comment_df, post_df = get_wsb_data()

    sentiment = comment_df.groupby("sentiment").count()["text"]
    grouped_df = comment_df.groupby("sentiment")["score"].sum().reset_index()
    grouped_sentiment = grouped_df['sentiment']
    grouped_score = grouped_df['score']


    # Create a dictionary to return
    return_dict = {}
    return_dict['total_sentiment'] = sentiment.to_dict()
    return_dict['sentiment'] = grouped_sentiment.to_dict()
    return_dict['score'] = grouped_score.to_dict()
    # return_dict['comment'] = """To make plots: plot x=sentiment, y=score for score of each sentiment by upvote
                                # plot x=sentiment, y=total_sentiment for number of comments in each sentiment class"""

    # Convert to dictionary and return
    return return_dict
