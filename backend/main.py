"""FastAPI app: loads the pickled Iris model and serves predictions."""
import pickle
from pathlib import Path

import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# 1. Load the model ONCE when the server starts
# ------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model" / "iris_model.pkl"
FRONTEND_PATH = Path(__file__).parent.parent /"frontend"/ "index.html"
if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Run the training notebook first."
    )

if not FRONTEND_PATH.exists():
    raise RuntimeError(
        f"Frontend not found at {FRONTEND_PATH}."
    )
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
FEATURES: list[str] = bundle["features"]
CLASS_NAMES: list[str] = bundle["class_names"]

if bundle.get("sklearn_version") != sklearn.__version__:
    print(
        f"⚠️  Model trained with scikit-learn {bundle.get('sklearn_version')}, "
        f"but running {sklearn.__version__}. Pin the same version in both projects."
    )

# ------------------------------------------------------------------
# 2. App + CORS (so the browser frontend is allowed to call us)
# ------------------------------------------------------------------
app = FastAPI(title="Iris Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production: ["https://your-frontend.com"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# 3. Schemas (Pydantic validates incoming JSON automatically)
# ------------------------------------------------------------------
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=20, examples=[5.1])
    sepal_width: float = Field(..., gt=0, lt=20, examples=[3.5])
    petal_length: float = Field(..., gt=0, lt=20, examples=[1.4])
    petal_width: float = Field(..., gt=0, lt=20, examples=[0.2])


class PredictionOut(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]

# ------------------------------------------------------------------
# 4. Routes
# ------------------------------------------------------------------
@app.get("/")
def root():
    return FileResponse(FRONTEND_PATH)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "features": FEATURES,
        "classes": CLASS_NAMES,
        "test_accuracy": bundle.get("test_accuracy"),
    }


@app.post("/predict", response_model=PredictionOut)
def predict(data: IrisInput):
    try:
        # 1-row DataFrame with the SAME column order used in training
        row = pd.DataFrame([data.model_dump()])[FEATURES]
        probs = model.predict_proba(row)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    best = int(probs.argmax())
    return PredictionOut(
        prediction=CLASS_NAMES[best],
        confidence=round(float(probs[best]), 4),
        probabilities={
            name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)
        },
    )