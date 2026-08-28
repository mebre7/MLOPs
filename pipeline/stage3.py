# Model Training
import pandas as pd
import pickle
import os
from sklearn.linear_model import ElasticNet

os.makedirs("models", exist_ok=True)
train = pd.read_csv("data/processed/train.csv")

train_x = train.drop(["quality"], axis=1)
train_y = train[["quality"]]

model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
model.fit(train_x, train_y)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Stage 3 complete: Model trained and saved.")