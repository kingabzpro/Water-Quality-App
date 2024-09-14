import os
import json
import skops.io as sio
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load preprocessed data
data = pd.read_csv("data/preprocessed_data.csv")
features = data.drop("Potability", axis=1)
target = data["Potability"]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)


# Train the model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


# Save the model using skops
os.makedirs("models", exist_ok=True)
sio.dump(model, "models/water_quality_model.skops")


# Track experiments and save metadata
metadata = {
    "model_name": "RandomForestClassifier",
    "parameters": model.get_params(),
    "training_score": model.score(X_train, y_train),
}


with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)