
# ML Pipeline Lite — No-Compile Runtime (FastAPI + NumPy)

This is a **Windows-friendly** end-to-end ML deployment: the API runs with **only NumPy** (no scikit-learn/SciPy at runtime).  
Training was done offline; the exported `models/model_params.json` contains:
- scaler means & scales for 30 features
- 3 quantile thresholds for a 4-bin engineered feature from `mean_radius`
- logistic regression weights + intercept


## Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# open http://127.0.0.1:8000/docs
```

## Predict
Use `example.json` and POST to `/predict`.

## Docker
```bash
docker build -t ml-pipeline-lite:latest .
docker run -p 8000:8000 ml-pipeline-lite:latest
```
