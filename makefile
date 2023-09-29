CACHE_FILE := .container_id_cache

run_api:
	uvicorn API.fastapi:app --reload

run_frontend:
	streamlit run frontend/index.py

update_packages:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

build_container_local:
	docker build -t api .
	docker run -p 8080:8000 -e REDDIT_CLIENT_ID -e REDDIT_CLIENT_SECRET -e REDDIT_SCRAPE_USERNAME -e REDDIT_SCRAPE_PASSWORD -e REDDIT_SCRAPE_AGENT -e REDDIT_SCRAPE_REDIRECT api

stop_container:
	CONTAINER_ID=$$(docker ps -q --filter "ancestor=api"); \
	echo "$$CONTAINER_ID" > $(CACHE_FILE); \
	docker stop $$CONTAINER_ID

start_container:
	CONTAINER_ID=$$(cat $(CACHE_FILE)); \
     	docker start $$CONTAINER_ID

update_container:
	CONTAINER_ID=$$(cat .container_id_cache); \
    	docker stop $$CONTAINER_ID && docker rm $$CONTAINER_ID || true
	$(MAKE) build_container_local
