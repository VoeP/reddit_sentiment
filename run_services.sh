#!/bin/bash

# Install packages
# We do this here because otherwise it makes the docker image fucking massive
pip install -r requirements.txt

# First run the print_credentials script which will verify that the environment variables are correctly configured
echo "Please verify these credentials are as expected otherwise data collection will not work"
python scripts/print_credentials.py

uvicorn API.fastapi:app --host 0.0.0.0 --port $PORT &
python scripts/scrape_reddit_data.py
