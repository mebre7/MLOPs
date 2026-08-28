# Data Ingestion
import pandas as pd
import os

# Simulating data download and saving locally
os.makedirs("data/raw", exist_ok=True)
df = pd.read_csv("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv", sep=";")
df.to_csv("data/raw/winequality-red.csv", index=False)
print("Stage 1 complete: Raw data saved.")