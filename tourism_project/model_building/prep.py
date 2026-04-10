import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

repo_id = "novicetopper/wellness-tourism-data"
repo_type = "dataset"
api = HfApi(token=os.getenv("HF_TOKEN"))

local_path = hf_hub_download(
    repo_id=repo_id,
    repo_type=repo_type,
    filename="tourism.csv",
    token=os.getenv("HF_TOKEN"),
)

df = pd.read_csv(local_path)
print("Loaded dataset from Hugging Face dataset space.")

# Remove unnecessary columns and clean the data
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

if df["ProdTaken"].dtype != int:
    df["ProdTaken"] = df["ProdTaken"].astype(int)

initial_shape = df.shape
df = df.dropna().reset_index(drop=True)
print(f"Dropped missing values: {initial_shape} -> {df.shape}")

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

os.makedirs("tourism_project/data", exist_ok=True)
X_train.to_csv("tourism_project/data/X_train.csv", index=False)
X_test.to_csv("tourism_project/data/X_test.csv", index=False)
y_train.to_csv("tourism_project/data/y_train.csv", index=False)
y_test.to_csv("tourism_project/data/y_test.csv", index=False)

for filename in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
    api.upload_file(
        path_or_fileobj=os.path.join("tourism_project/data", filename),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Uploaded {filename} to Hugging Face dataset space.")
