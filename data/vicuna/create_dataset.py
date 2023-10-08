import os
import requests
import csv
import json
from tqdm import tqdm


JSONL_FILE = "question.jsonl"
JSONL_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/"
    "833d65032a715240a3978f4a8f08e7a496c83cb1/fastchat/eval/table/"
    "question.jsonl"
)
DATA_FILE = "data.csv"


def download(url, file):
    response = requests.get(url)
    with open(file, "wb") as f:
        f.write(response.content)


def save(file, request):
    with open(file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([request])


if __name__ == "__main__":
    if not os.path.exists(JSONL_FILE):
        download(JSONL_URL, JSONL_FILE)

    save(DATA_FILE, "request")
    with open(JSONL_FILE) as f:
        for line in tqdm(f):
            request = line.strip()
            request = json.loads(request)["text"]
            save(DATA_FILE, request)
