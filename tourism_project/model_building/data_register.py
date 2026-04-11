from huggingface_hub import HfApi, create_repo
import os

repo_id = "novicetopper/wellness-tourism-data"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists.")
except Exception:
    print(f"Creating dataset repository '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print("Dataset repository created.")

local_file = "../data/tourism.csv"
api.upload_file(
    path_or_fileobj=local_file,
    path_in_repo="tourism.csv",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Uploaded tourism.csv to the Hugging Face dataset space.")
