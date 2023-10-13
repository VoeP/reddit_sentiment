from time import sleep
import time
import os
import traceback
import datetime
import reddit_sentiment_modules.scraping_python_functions as spf

def connect():
    """Establishes a connection to reddit using praw."""
    try:
        return spf.init_reddit()
    except:
        print("Failed to connect to reddit")
        return None

def get_comments_df(reddit):
    comment_details = spf.get_comments_from_hot_recursive(reddit, num=20, subreddit="wallstreetbets", max_depth=None, verbose=1)
    print("Scraped comments")
    now = time.time()
    comment_details = spf.process_data(comment_details)
    dt = time.time() - now
    print(f"Processed comments in {round(dt)} seconds")
    return comment_details

def get_posts_df(reddit):
    post_details = spf.get_post_info(reddit)
    print("Scraped post details")
    now = time.time()
    post_details = spf.process_data(post_details, text_column="titles")
    dt = time.time() - now
    print(f"Processed post details in {round(dt)} seconds")
    return post_details

def main():
    reddit = connect()

    # Make the reddit_data folder if it doesn't exist
    try:
        os.mkdir("reddit_data")
        print("Created reddit_data folder")
    except:
        pass

    try:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        # Get the comments dataframe
        df_comments = get_comments_df(reddit)
        # Save it
        df_comments.to_csv(f"reddit_data/reddit_comments_{date}.csv")
        print("Saved comments")

        # Same with the posts dataframe
        df_posts = get_posts_df(reddit)
        df_posts.to_csv(f"reddit_data/reddit_posts_{date}.csv")
        print("Saved posts")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print(traceback.format_exc())

    print("Finished scraping job")


if __name__ == "__main__":
    # Initial run to get some data
    main()
