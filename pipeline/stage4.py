# Model Evaluation
import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

test = pd.read_csv("data/processed/test.csv")
test_x = test.drop(["quality"], axis=1)
test_y = test[["quality"]]

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

preds = model.predict(test_x)
rmse = np.sqrt(mean_squared_error(test_y, preds))
r2score = r2_score(test_y, preds)

metrics = {
    "rmse": rmse,
    "r2_score": r2score
}
os.makedirs("reports", exist_ok=True)
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Stage 4 complete: Evaluation metrics save (RMSE: {rmse}, R2 Score: {r2score}).")