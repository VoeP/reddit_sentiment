CACHE_FILE := .container_id_cache

run_api:
	uvicorn API.fastapi:app --reload --host localhost --port 8000

run_datacollection_api:
	uvicorn data_collection.data_collection_api:app --reload --host localhost --port 8000

run_frontend:
	streamlit run frontend/index.py

update_packages:
	pip install --upgrade pip
	pip install -r requirements1.txt
	pip install -r requirements2.txt
	pip install -r requirements3.txt
	pip install -e .

build_container_api:
	mv dockerfile-api dockerfile
	-docker build -t $$GCR_MULTI_REGION/$$GCP_PROJECT_ID/$$DOCKER_IMAGE_NAME .
	mv dockerfile dockerfile-api

push_container_api:
	docker push $$GCR_MULTI_REGION/$$GCP_PROJECT_ID/$$DOCKER_IMAGE_NAME

deploy_container_api:
	gcloud run deploy --image $$GCR_MULTI_REGION/$$GCP_PROJECT_ID/$$DOCKER_IMAGE_NAME --platform managed --region $$GCR_REGION

build_data_collection_container:
	mv dockerfile-datacollection dockerfile
	# Use the service account key to authenticate the container to GCP
	cp /home/ed/code/Eatkin/gcp/reddit-sentiment-400608-24bc7c65b16e.json .
	-docker build -t $$GCR_MULTI_REGION/$$GCP_PROJECT_ID/data_collection .
	mv dockerfile dockerfile-datacollection
	rm reddit-sentiment-400608-24bc7c65b16e.json

push_data_collection_container:
	docker push $$GCR_MULTI_REGION/$$GCP_PROJECT_ID/data_collection

deploy_data_collection_container:
	gcloud run deploy --image $$GCR_MULTI_REGION/$$GCP_PROJECT_ID/data_collection --platform managed --region $$GCR_REGION --update-env-vars REDDIT_CLIENT_ID="$$REDDIT_CLIENT_ID",REDDIT_CLIENT_SECRET="$$REDDIT_CLIENT_SECRET",REDDIT_SCRAPE_USERNAME="$$REDDIT_SCRAPE_USERNAME",REDDIT_SCRAPE_PASSWORD="$$REDDIT_SCRAPE_PASSWORD",REDDIT_SCRAPE_AGENT="$$REDDIT_SCRAPE_AGENT",REDDIT_SCRAPE_REDIRECT="$$REDDIT_SCRAPE_REDIRECT"
