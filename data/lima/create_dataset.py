import os
import csv
import json
from tqdm import tqdm
from huggingface_hub import login, hf_hub_download


JSONL_FILE = "train.jsonl"
JSONL_URL = "GAIR/lima"
DATA_FILE = "data.csv"

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")


def download(url, file, access_token):
    login(access_token)
    hf_hub_download(repo_id=url, filename=file, repo_type="dataset", local_dir=".")


def save(file, request):
    with open(file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([request])


if __name__ == "__main__":
    if not os.path.exists(JSONL_FILE):
        download(JSONL_URL, JSONL_FILE, ACCESS_TOKEN)

    save(DATA_FILE, "request")
    with open(JSONL_FILE) as f:
        for line in tqdm(f):
            request = line.strip()
            request = json.loads(request)["conversations"][0]

            save(DATA_FILE, request)
