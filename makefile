CACHE_FILE := .container_id_cache

run_api:
	uvicorn API.fastapi:app --reload

run_frontend:
	streamlit run frontend/index.py

update_packages:
	pip install --upgrade pip
	pip install -r requirements.txt

build_container_local:
	docker build -t api .
	docker run -p 8080:8000 api

stop_container:
	CONTAINER_ID=$$(docker ps -q --filter "ancestor=api"); \
	echo "$$CONTAINER_ID" > $(CACHE_FILE); \
	docker stop $$CONTAINER_ID

start_container:
	CONTAINER_ID=$$(cat $(CACHE_FILE)); \
     	docker start $$CONTAINER_ID
