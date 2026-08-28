# Preprocessing and Splitting
import pandas as pd
import os
from sklearn.model_selection import train_test_split
os.makedirs("data/processed", exist_ok=True)
df = pd.read_csv("data/raw/winequality-red.csv")

train, test = train_test_split(df, test_size=0.25, random_state=42)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
print("Stage 2 complete: Data processes and split.")