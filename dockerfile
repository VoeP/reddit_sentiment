# Python image
FROM python:3.10.6-buster

# Copy API, requirements and the makefile
# Syntax COPY [source] [destination]
COPY API /API
COPY requirements.txt /requirements.txt
COPY reddit_sentiment_modules/*.py /reddit_sentiment_modules/
COPY scripts/scrape_reddit_data.py /scripts/scrape_reddit_data.py
COPY scripts/print_credentials.py /scripts/print_credentials.py
# Grab any pre-existing data
COPY reddit_data /reddit_data
COPY run_services.sh /run_services.sh

# Upgrade pip
RUN pip install --upgrade pip
# Install packages
RUN pip install -r requirements.txt
# Move setup.py to root because of stupid Docker reasons
RUN mv reddit_sentiment_modules/setup.py .
RUN pip install .

# Make run services executable
RUN chmod +x run_services.sh

# Run it
# IMPORTANT!! This is where the code for hosting the API runs from
CMD ["./run_services.sh"]
