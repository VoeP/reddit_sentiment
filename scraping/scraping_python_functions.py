import praw
from reddit_credentials import *

def init_reddit():
    reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    password=password,
    user_agent=user_agent,
    redirect_uri=redirect_uri,
    username=user_name,
    )
    return reddit

def get_comments_from_hot(reddit):
    subreddit = reddit.subreddit("wallstreetbets")
    text=[]
    score=[]
    level=[]
    for i in subreddit.hot(limit=20):
        submission=reddit.submission(i)
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
