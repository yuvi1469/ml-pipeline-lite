
from fastapi import FastAPI, HTTPException
from .schema import PredictRequest, PredictResponse
import numpy as np, json, os

app = FastAPI(title="Breast Cancer Classifier API (Lite)", version="1.0.0")

PARAMS_PATH = os.getenv("PARAMS_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "model_params.json"))
PARAMS_PATH = os.path.abspath(PARAMS_PATH)

with open(PARAMS_PATH, "r") as f:
    P = json.load(f)

FEATURE_NAMES = P["feature_names"]
MEAN = np.array(P["scaler_mean"], dtype=float)
SCALE = np.array(P["scaler_scale"], dtype=float)
EDGES = np.array(P["bin_edges_mean_radius"], dtype=float)  # length 3
W = np.array(P["coef"], dtype=float).reshape(-1)  # length 34
B = float(P["intercept"])
THRESH = float(P["threshold"])

def one_hot_bin(mr: float) -> np.ndarray:
    if mr < EDGES[0]: idx = 0
    elif mr < EDGES[1]: idx = 1
    elif mr < EDGES[2]: idx = 2
    else: idx = 3
    v = np.zeros(4, dtype=float); v[idx] = 1.0
    return v

def transform(x30: np.ndarray) -> np.ndarray:
    z = (x30 - MEAN) / SCALE
    mr = float(x30[FEATURE_NAMES.index("mean_radius")])
    oh = one_hot_bin(mr)
    return np.concatenate([z, oh], axis=0)

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != 30:
        raise HTTPException(status_code=400, detail=f"Expected 30 features, got {len(req.features)}")
    x = np.array(req.features, dtype=float).reshape(-1)
    X = transform(x)
    logit = float(np.dot(W, X) + B)
    proba = float(sigmoid(logit))
    pred = int(proba >= THRESH)
    return PredictResponse(probability=proba, prediction=pred)
