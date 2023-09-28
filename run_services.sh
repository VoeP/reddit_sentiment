#!/bin/bash

uvicorn API.fastapi:app --host 0.0.0.0 &
python scripts/scrape_reddit_data.py
