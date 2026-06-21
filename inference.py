import json
import os
import joblib
import numpy as np
import pandas as pd

JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"
CLASS_NAMES = ["iris-setosa", "iris-versicolor", "iris-virginica"]
FEATURE_NAMES = ["sepal_length","sepal_width","petal_length","petal_width"]

def model_fn(model_dir: str):
    return joblib.load(os.path.join(model_dir, "model_iris.joblib"))

def input_fn(request_body, request_content_type: str) -> pd.DataFrame:

    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        instances = payload["instances"]
        return pd.DataFrame(instances, columns=FEATURE_NAMES)

    if request_content_type == CSV_CONTENT_TYPE:
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")
        rows = [[float(x) for x in line.split(",")]
            for line in request_body.strip().splitlines()
            if line.strip()]
        return pd.DataFrame(rows, columns=FEATURE_NAMES)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: pd.DataFrame, pipeline) -> dict:
    probs = pipeline.predict_proba(input_data)
    class_ids = np.argmax(probs, axis=1)
    labels = [CLASS_NAMES[int(i)] for i in class_ids]
    return {"probabilities": probs.tolist(),
            "predictions": class_ids.tolist(),
            "labels": labels,}

def output_fn(prediction: dict, accept_content_type: str):
    if accept_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")
