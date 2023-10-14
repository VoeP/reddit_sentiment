import praw
from reddit_sentiment_modules.credentials import Credentials
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


def init_reddit():
    """Initializes a connection to reddit with praw using the credentials defined in the reddit_credentials.py file.
    If you don't have these you can ask VoeP for them or define them for yourself, creating them is definitiely not difficult
    whatsoever."""
    # Create the credentials object
    cred = Credentials()
    reddit = praw.Reddit(
    client_id=cred.client_id,
    client_secret=cred.client_secret,
    password=cred.password,
    user_agent=cred.agent,
    redirect_uri=cred.redirect_uri,
    username=cred.username,
    )
    return reddit


def get_comments_from_hot(reddit, num=20, subreddit="wallstreetbets"):
    """Gets the comment dataset from num first posts in the hot category from subreddit. reddit is the
    connection specified with praw, which can be initialized using the init_reddit function."""
    subreddit = reddit.subreddit(subreddit)
    text=[]
    score=[]
    level=[]
    posts=[]
    urls = []
    for i in subreddit.hot(limit=num):
        submission=reddit.submission(i)
        url = submission.url
        for top_level_comment in submission.comments:
            try:
                text.append(top_level_comment.body)
                score.append(top_level_comment.score)
                level.append("top")
                posts.append(submission.title)
                urls.append(url)
                for second_level_comment in top_level_comment.replies:
                    text.append(second_level_comment.body)
                    score.append(second_level_comment.score)
                    level.append("second")
                    posts.append(submission.title)
                    urls.append(url)
            except AttributeError:
                pass
    df_comments = pd.DataFrame.from_dict({"text":text, "score":score, "level":level,"post": posts})
    return df_comments

def get_comments_from_hot_recursive(reddit, num=20, subreddit="wallstreetbets", max_depth=None, verbose=False):
    """Gets the comment data set and recursively crawls the comments"""
    def get_comments(submission, url, title, depth=1):
        thread = submission.comments if depth == 1 else submission.replies
        # Don't exceed max_depth
        if max_depth is not None and depth == max_depth:
            return

        if verbose >= 2:
            print(f"Scraping comments from {url} at depth {depth}")

        for comment in thread:
            try:
                text.append(comment.body)
                score.append(comment.score)
                level.append(depth)
                posts.append(title)
                urls.append(url)
                get_comments(comment, url, title, depth+1)
            except AttributeError:
                pass

    subreddit = reddit.subreddit(subreddit)
    text = []
    score = []
    level = []
    posts = []
    urls = []
    for i in subreddit.hot(limit=num):
        submission = reddit.submission(i)
        title = submission.title
        url = submission.permalink
        get_comments(submission, url, title)

    if verbose >= 1:
        print(f"Collected {len(text)} comments")
        print(f"Max depth: {max(level)}")

    df_comments = pd.DataFrame.from_dict({"text": text, "score": score, "level": level, "post": posts, "url": urls})
    return df_comments


def get_post_info(reddit, num=20, subreddit="wallstreetbets"):
    """Gets the post detail dataset from num first posts in the hot category from subreddit. reddit is the
    connection specified with praw, which can be initialized using the init_reddit function."""
    subreddit = reddit.subreddit(subreddit)
    ids=[]
    bodies=[]
    original_content_flag=[]
    num_comments=[]
    titles=[]
    scores=[]
    controversiality=[]
    urls = []
    for i in subreddit.hot(limit=num):
        submission=reddit.submission(i)
        ids.append(str(submission.id))
        bodies.append(''+str(submission.selftext))
        original_content_flag.append(str(submission.is_original_content))
        num_comments.append(str(submission.num_comments))
        titles.append(str(submission.title))
        scores.append(str(submission.score))
        controversiality.append(str(submission.upvote_ratio))
        urls.append(submission.permalink)

    post_df=pd.DataFrame.from_dict({"ids": ids, "titles": titles, "scores": scores, "controversiality": controversiality,
                                    "bodies": bodies, "original":original_content_flag, "num_comments": num_comments,
                                    "url": urls})
    return post_df



def get_comments_for_one_submission(submission):
    """Takes a submission object from praw and gets the comment dataset for it."""
    text=[]
    score=[]
    level=[]
    for top_level_comment in submission.comments:
            try:
                text.append(top_level_comment.body)
                score.append(top_level_comment.score)
                level.append("top")
                for second_level_comment in top_level_comment.replies:
                    text.append(second_level_comment.body)
                    score.append(second_level_comment.score)
                    level.append("second")
            except AttributeError:
                pass
    df_comments = pd.DataFrame.from_dict({"text":text, "score":score, "level":level})
    return df_comments


def process_data(data, text_column="text", tokenizer1=None, model1=None, tokenizer2=None, model2=None):
    """Takes in a dataframe from the get_comments_* series of functions and calculates sentiment and emotions
    for this dataframe based on the text_column argument. tokenizer & model 1 are for defining what sentiment model to use,
    nlptown/bert-base-multilingual-uncased-sentiment being the default, and tokenizer & model 2 are for defining the
    emotion model, cardiffnlp/twitter-roberta-base-emotion being the default. The models need to have the same output
    format as the default mdoels to work."""

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


    def sentiment_score(review):
        return pipeline_bert(review)[0]

    data['sentiment'] = data[text_column].apply(lambda x: sentiment_score(x[:512]))

    def get_emotion_scores_as_dict(comments:list)->dict:
        joy = []
        optimism = []
        anger = []
        sadness = []
        emotions= [anger, joy, optimism, sadness]
        for comment in comments:
            encoded_input = emotions_tokeniser(comment, return_tensors='pt')
            output = emotions_model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            for i, score in enumerate(scores):
                emotions[i].append(score)

        return {"joy": joy ,"optimism": optimism ,"anger": anger ,"sadness": sadness}

    df_emotions = pd.DataFrame.from_dict(get_emotion_scores_as_dict(data[text_column]))

    df_emotions[df_emotions<0]=0
    data=pd.concat([data, df_emotions], axis=1)

    return data

def plot_emotions(df):
    """Takes in a dataframe that has a 'joy', 'optimism', 'anger' and 'sadness' column and makes plots based on these"""

    sum_joy = df["joy"].sum()
    sum_optimism = df["optimism"].sum()
    sum_anger = df["anger"].sum()
    sum_sadness = df["sadness"].sum()

    keys=["joy", "optimism", "anger", "sadness"]
    data=[sum_joy,
        sum_optimism,
        sum_anger,
        sum_sadness]

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, position 1
    plt.pie(data, labels=keys, autopct='%1.1f%%')
    plt.title('Emotional pie')
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, position 2
    plt.bar(keys, data)
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Emotion bars')
    plt.tight_layout()  # Automatically adjusts subplot parameters to give specified padding
    plt.savefig("emotions.png")


def sentiment_barplots(df):
    """Takes in a dataframe that has the 'sentiment' column and a 'scores' column and makes plots based on these"""
    sents=df.groupby("sentiment").count()["text"]
    grouped_df = df.groupby("sentiment")["score"].sum().reset_index()
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, position 1
    plt.bar(grouped_df["sentiment"], grouped_df["score"])
    plt.title('Score of each sentiment by upvotes')
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, position 2
    plt.bar(grouped_df["sentiment"], sents)
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Number fo comments in each sentiment class')
    plt.tight_layout()  # Automatically adjusts subplot parameters to give specified padding
    plt.savefig("sentiments.png")
