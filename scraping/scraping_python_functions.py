import praw
from reddit_credentials import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def init_reddit():
    """Initializes a connection to reddit with praw using the credentials defined in the reddit_credentials.py file.
    If you don't have these you can ask VoeP for them or define them for yourself, creating them is definitiely not difficult
    whatsoever."""
    reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    password=password,
    user_agent=user_agent,
    redirect_uri=redirect_uri,
    username=user_name,
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
    for i in subreddit.hot(limit=num):
        submission=reddit.submission(i)
        for top_level_comment in submission.comments:
            try:
                text.append(top_level_comment.body)
                score.append(top_level_comment.score)
                level.append("top")
                posts.append(submission.title)
                for second_level_comment in top_level_comment.replies:
                    text.append(second_level_comment.body)
                    score.append(second_level_comment.score)
                    level.append("second")
                    posts.append(submission.title)
            except AttributeError:
                pass
    df_comments = pd.DataFrame.from_dict({"text":text, "score":score, "level":level,"post": posts})
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
    for i in subreddit.hot(limit=num):
        submission=reddit.submission(i)
        ids.append(str(submission.id))
        bodies.append(''+str(submission.selftext))
        original_content_flag.append(str(submission.is_original_content))
        num_comments.append(str(submission.num_comments))
        titles.append(str(submission.title))
        scores.append(str(submission.score))
        controversiality.append(str(submission.upvote_ratio))
    breakpoint()

    post_df=pd.DataFrame.from_dict({"ids":ids, "titles":titles, "scores": scores, "controversiality":controversiality,
                                    "bodies": bodies, "original":original_content_flag, "num_comments": num_comments})
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

    if tokenizer1==None:
        tokenizer1=AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    if model1==None:
        model1=AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    if tokenizer2==None:
        tokenizer2 = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
    if model2==None:
        model2=RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")


    def sentiment_score(review):
        tokens = tokenizer1.encode(review, return_tensors='pt')
        result = model1(tokens)
        return int(torch.argmax(result.logits))+1


    data['sentiment'] = data[text_column].apply(lambda x: sentiment_score(x[:512]))

    def yield_raw_predictions(comment:str)->str:
        inputs = tokenizer2(comment, return_tensors="pt")
        with torch.no_grad():
            logits = model2(**inputs).logits
        return logits.tolist()[0]


    def get_emotion_scores_as_dict(comments:list)->dict:
        joy = []
        optimism = []
        anger = []
        sadness = []
        emotions= [joy, optimism, anger, sadness]
        for comment in comments:
            raw_predictions=yield_raw_predictions(comment[:512])
            for i in np.arange(len(raw_predictions)):
                emotions[i].append(raw_predictions[i])

        return {"joy": joy ,"optimism": optimism ,"anger": anger ,"sadness": sadness}

    df_emotions = pd.DataFrame.from_dict(get_emotion_scores_as_dict(data[text_column]))

    df_emotions[df_emotions<0]=0
    data=pd.concat([data, df_emotions], axis=1)

    return data

def save_current_hot_data_as_csvs(path_to_storage:str, reddit=init_reddit()):
    """This function saves both the comment and post csvs to a specified location 'path_to_storage'"""
    df_comments=get_comments_from_hot(reddit)
    df_comments=processed_data=process_data(df_comments)
    df_comments.to_csv(f"{path_to_storage}/comment_data.csv")
    df_posts=get_post_info(reddit)
    df_posts=process_data(df_posts, text_column="titles")
    df_posts.to_csv(f"{path_to_storage}/post_data.csv")




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
