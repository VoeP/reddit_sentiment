import os
from fastapi import FastAPI
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import pandas as pd

# Create the app object
app = FastAPI()

# Define constants
CHUNK_SIZE = 512

# Generic error function to use for all endpoints
def message_error(message):
    if message is None:
        return {"error": "No comment provided"}
    return None

# Generic function for loading a huggingface model
def load_huggingface_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return TextClassificationPipeline(model=model, tokenizer=tokenizer)

# Generic function for getting tokenizer and model from huggingface
def load_huggingface_tokenizer_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return tokenizer, model


pipeline_bert = load_huggingface_model("nlptown/bert-base-multilingual-uncased-sentiment")
emotions_tokeniser, emotions_model = load_huggingface_tokenizer_model("cardiffnlp/twitter-roberta-base-emotion")

def get_wsb_data():
    # Find the local csv files to process
    our_path = os.path.abspath(os.path.dirname(__file__))
    par_dir = os.path.dirname(our_path)
    # csvs are in data_for_plotting
    csv_path = os.path.join(our_path, par_dir, "reddit_data")

    # Find all csvs in the folder
    all_csvs = os.listdir(csv_path)
    # they're of the format reddit_comments_yyyy_mm_dd.csv and reddit_posts_yyyy_mm_dd.csv so filter and sort
    comments_csvs = sorted([csv for csv in all_csvs if "comments" in csv])
    posts_csvs = sorted([csv for csv in all_csvs if "posts" in csv])
    comment_df = pd.read_csv(os.path.join(csv_path, comments_csvs[-1]))
    post_df = pd.read_csv(os.path.join(csv_path, posts_csvs[-1]))
    return comment_df, post_df

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
