# Python image
FROM python:3.10.6-buster

# Copy API, requirements and the makefile
# Syntax COPY [source] [destination]
COPY API /API
COPY requirements.txt /requirements.txt
COPY makefile /makefile
COPY reddit_sentiment_modules /reddit_sentiment_modules
COPY scripts/scrape_reddit_data.py /scripts/scrape_reddit_data.py
# Grab any pre-existing data
COPY reddit_data /reddit_data
COPY run_services.sh /run_services.sh

# Run the makefile to install requirements (upgrades pip and install requirements)
RUN make update_packages

# Make run services executable
RUN chmod +x run_services.sh

# Run it
# IMPORTANT!! This is where the code for hosting the API runs from
CMD ["./run_services.sh"]
