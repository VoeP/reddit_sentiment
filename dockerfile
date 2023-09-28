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

# Set environment variables
# These MUST BE SET in your zshrc file if you are building this container locally or it won't be able to collect data!
ENV REDDIT_CLIENT_ID=$REDDIT_CLIENT_ID
ENV REDDIT_CLIENT_SECRET=$REDDIT_CLIENT_SECRET
ENV REDDIT_SCRAPE_USERNAME=$REDDIT_SCRAPE_USERNAME
ENV REDDIT_SCRAPE_PASSWORD=$REDDIT_SCRAPE_PASSWORD
ENV REDDIT_SCRAPE_AGENT=$REDDIT_SCRAPE_AGENT
ENV REDDIT_SCRAPE_REDIRECT=$REDDIT_SCRAPE_REDIRECT

# Run it
# IMPORTANT!! This is where the code for hosting the API runs from
CMD ["./run_services.sh"]
