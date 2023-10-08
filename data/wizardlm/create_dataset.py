import os
import requests
import csv
import json
from tqdm import tqdm


JSONL_FILE = "WizardLM_testset.jsonl"
JSONL_URL = (
    "https://raw.githubusercontent.com/nlpxucan/WizardLM/"
    "ce0d433589335d419aa8101abd71685ae7d187f3/WizardLM/data/WizardLM_testset.jsonl"
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
            request = json.loads(request)["Instruction"]
            save(DATA_FILE, request)
