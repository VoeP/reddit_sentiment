run_api:
	uvicorn API.fastapi:app --reload

run_frontend:
	streamlit run frontend/index.py
