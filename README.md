# 🌸 Iris Classifier: Jupyter → Pickle → FastAPI

Train a scikit-learn model in a notebook, pickle it, and serve predictions
through FastAPI with a simple HTML frontend.

## Project structure
```
notebook/   → trains the model and saves backend/model/iris_model.pkl
backend/    → FastAPI app (serves the API and the frontend)
frontend/   → HTML form that calls /predict
```

## Run locally (requires [uv](https://docs.astral.sh/uv/))
```bash
# 1. Train (optional, the .pkl is already committed)
cd notebook
uv sync
uv run jupyter lab        # run all cells

# 2. Start the app
cd ../backend
uv sync
uv run uvicorn main:app --reload
```
Open http://127.0.0.1:8000 for the app, or http://127.0.0.1:8000/docs for the API docs.

## API
`POST /predict`
```json
{"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
```
→ `{"prediction": "virginica", "confidence": 0.99, ...}`