import time, datetime
from fastapi import FastAPI
from reddit_sentiment_modules import scraping_python_functions as spf
import pandas as pd
from google.cloud import bigquery

# Create the app object
app = FastAPI()


# Set up definitions
def connect():
    """Establishes a connection to reddit using praw."""
    try:
        return spf.init_reddit()
    except:
        print("Failed to connect to reddit")

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

def push_to_bigquery(client, comments, posts):
    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # Constants
    project_id = "reddit-sentiment-400608"
    dataset_id = "wallstreetbets"
    table_id = "reddit_comments"

    comments_schema = [
        bigquery.SchemaField("date", "DATE"),
        bigquery.SchemaField("text", "STRING"),
        bigquery.SchemaField("score", "INTEGER"),
        bigquery.SchemaField("level", "INTEGER"),
        bigquery.SchemaField("post", "STRING"),
        bigquery.SchemaField("url", "STRING"),
        bigquery.SchemaField("sentiment", "FLOAT"),
        bigquery.SchemaField("joy", "FLOAT"),
        bigquery.SchemaField("optimism", "FLOAT"),
        bigquery.SchemaField("anger", "FLOAT"),
        bigquery.SchemaField("sadness", "FLOAT"),
    ]

    posts_schema = [
        bigquery.SchemaField("date", "DATE"),
        bigquery.SchemaField("ids", "STRING"),
        bigquery.SchemaField("titles", "STRING"),
        bigquery.SchemaField("controversiality", "FLOAT"),
        bigquery.SchemaField("bodies", "STRING"),
        bigquery.SchemaField("original", "BOOLEAN"),
        bigquery.SchemaField("num_comments", "INTEGER"),
        bigquery.SchemaField("url", "STRING"),
        bigquery.SchemaField("sentiment", "FLOAT"),
        bigquery.SchemaField("joy", "FLOAT"),
        bigquery.SchemaField("optimism", "FLOAT"),
        bigquery.SchemaField("anger", "FLOAT"),
        bigquery.SchemaField("sadness", "FLOAT"),
    ]

    # Set up job configs
    comments_job_config = bigquery.LoadJobConfig(
        schema=comments_schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    posts_job_config = bigquery.LoadJobConfig(
        schema=posts_schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )

    # Set date column to today's date
    comments["date"] = todays_date
    posts["date"] = todays_date

    # Bigquery requires dataframe to be in a list so convert to records format
    comments_records = comments.to_dict(orient="records")
    posts_records = posts.to_dict(orient="records")

    # Empty bodies appear as NaN and thus as floats so we need to convert them to empty strings
    for record in posts_records:
        if type(record['bodies']) == float:
            record['bodies'] = ''

    print("Pushing comments to Bigquery")
    client.load_table_from_json(
        comments_records, f"{project_id}.{dataset_id}.{table_id}", job_config=comments_job_config
    ).result()

    # Switch table id to posts
    table_id = "reddit_posts"

    print("Pushing posts to Bigquery")
    client.load_table_from_json(
        posts_records, f"{project_id}.{dataset_id}.{table_id}", job_config=posts_job_config
    ).result()


# Create the root endpoint
@app.get("/")
def index():
    return {"message": "Hello, World!"}

# Endpoint that can be poked to trigger data collection
@app.get("/get_data")
def get_data():
    # Constants
    project_id = "reddit-sentiment-400608"
    dataset_id = "wallstreetbets"
    table_id = "reddit_comments"

    try:
        # Establish connection
        reddit = connect()
        print("Reddit connection established")

        # Bigquery
        client = bigquery.Client(project=project_id)
        print("Bigquery connection established")
        try:
            # First thing we'll do is query the database to see if we have any data for today
            query = f"""
            SELECT MAX(date) as max_date
            FROM {project_id}.{dataset_id}.{table_id}
            """

            today = datetime.datetime.now().strftime("%Y-%m-%d")
            # This will return nothing if there is no data so we need to check for that
            query_df = client.query(query).to_dataframe()
            if not query_df.empty and query_df["max_date"].iloc[0] == today:
                return {"message": "Data collection already completed today"}
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            print("We couldn't check if data collection has already been completed today")
            print("This probably means that there is no data because it's the first time we've run this script")

        # Get the comments dataframe
        df_comments = get_comments_df(reddit)
        print("Comments dataframe created")

        # Same with the posts dataframe
        df_posts = get_posts_df(reddit)
        print("Posts dataframe created")

        # Now let's send them to Bigquery
        print("Sending data to Bigquery")
        try:
            push_to_bigquery(client, df_comments, df_posts)
            print("Data sent to Bigquery")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            print("Data not sent to Bigquery")

        return {"message": "Data collection successful"}
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return {"message": "Data collection failed"}
