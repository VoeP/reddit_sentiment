#!/bin/bash

# First run the print_credentials script which will verify that the environment variables are correctly configured
echo "Please verify these credentials are as expected otherwise data collection will not work"
python scripts/print_credentials.py

uvicorn API.fastapi:app --host 0.0.0.0 &
python scripts/scrape_reddit_data.py
