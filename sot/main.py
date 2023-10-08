import argparse
import os
import csv
import json
import shutil
import logging
from tqdm import tqdm
import pandas as pd

from models import get_model_class_from_name
from schedulers import get_scheduler_class_from_name
from utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--scheduler", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--output-data-filename", type=str, default="data.csv")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically answer yes to all user prompts.",
    )
    args, other_args = parser.parse_known_args()

    model_class = get_model_class_from_name(args.model)
    model, other_args = model_class.from_command_line_args(other_args)

    scheduler_class = get_scheduler_class_from_name(args.scheduler)
    scheduler, other_args = scheduler_class.from_command_line_args(
        other_args, model=model
    )

    if other_args != []:
        raise ValueError("Unknown arguments: {}".format(other_args))

    return args, model, scheduler


def save(file, content):
    for i in range(len(content)):
        if type(content[i]) != str:
            content[i] = json.dumps(content[i])
    with open(file, "a", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(content)


def main():
    args, model, scheduler = parse_args()
    output_data_path = os.path.join(args.output_folder, args.output_data_filename)

    skip_questions = set([])
    if os.path.exists(args.output_folder):
        if args.yes or (
            input(
                "Output folder {} already exists. Override (Y/N)? ".format(
                    args.output_folder
                )
            ).lower()
            in {"yes", "y"}
        ):
            print("Removing the existing output folder...")
            shutil.rmtree(args.output_folder)
        else:
            if os.path.exists(output_data_path):
                logging.info("Will skip existing results")
                skip_questions = pd.read_csv(output_data_path, usecols=["request"])
                skip_questions = set(skip_questions["request"].tolist())

    os.makedirs(args.output_folder, exist_ok=True)
    setup_logging(os.path.join(args.output_folder, "log.log"))

    scheduler.print_info()

    input_data = pd.read_csv(args.data_path, header=0, names=["request"])
    keys = None
    for _, row in tqdm(input_data.iterrows()):
        request = row["request"]
        if request in skip_questions:
            continue
        response = scheduler.get_response(request)
        if keys is None:
            keys = list(sorted(response.keys()))
            if not os.path.exists(output_data_path):
                save(output_data_path, keys)
        else:
            assert keys == list(sorted(response.keys()))
        response["request"] = request
        save(output_data_path, [response[k] for k in keys])


if __name__ == "__main__":
    main()
