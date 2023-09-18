import os

class Credentials:
    def __init__(self):
        self.client_id = os.environ.get('REDDIT_CLIENT_ID')
        self.client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
        self.username = os.environ.get('REDDIT_SCRAPE_USERNAME')
        self.password = os.environ.get('REDDIT_SCRAPE_PASSWORD')
        self.agent = os.environ.get('REDDIT_SCRAPE_AGENT')
        self.redirect_uri = os.environ.get('REDDIT_SCRAPE_REDIRECT')
