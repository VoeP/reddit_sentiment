from reddit_sentiment_modules.credentials import Credentials

'''Simple script to print credentials
Can be used to verify environment variables are set correctly'''

credentials = Credentials()
print("Client ID:", credentials.client_id)
print("Client Secret:", credentials.client_secret)
print("Username:", credentials.username)
print("Password:", credentials.password)
print("Agent:", credentials.agent)
print("Redirect URI:", credentials.redirect_uri)
