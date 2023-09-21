# Python image
FROM python:3.10.6-buster

# Copy API, requirements and the makefile
# Syntax COPY [source] [destination]
COPY API /API
COPY requirements.txt /requirements.txt
COPY makefile /makefile
COPY reddit_sentiment_modules /reddit_sentiment_modules

# No model yet but when we have one we will copy the model at this point
# COPY models/blah /model

# Run the makefile to install requirements (upgrades pip and install requirements)
RUN make update_packages

# Start the API (locally)
CMD uvicorn API.fastapi:app --host 0.0.0.0

# Start the API (once on GCP)
# CMD uvicorn API.fastapi:app --host 0.0.0.0 --port $PORT
