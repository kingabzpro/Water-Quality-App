import os

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Download the dataset
os.system("kaggle datasets download -d adityakadiwal/water-potability -p data --unzip")