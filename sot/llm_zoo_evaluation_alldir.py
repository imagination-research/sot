import argparse
import os
import csv
import json
from tqdm import tqdm
import pandas as pd
import logging
import itertools

from models import get_model_class_from_name
from utils.logging import setup_logging
from evaluation.llm_evaluation import llm_zoo_evaluation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--answer-file", type=str, required=True, action="append")
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument(
        "--output-data-filename", type=str, default="llm_zoo_evaluation.csv"
    )
    parser.add_argument(
        "--log-filename", type=str, default="llm_zoo_evaluation_log.log"
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically answer yes to all user prompts.",
    )
    args, other_args = parser.parse_known_args()

    model_class = get_model_class_from_name(args.model)
    model, other_args = model_class.from_command_line_args(other_args)

    if other_args != []:
        raise ValueError("Unknown arguments: {}".format(other_args))

    return args, model


def save(file, content):
    for i in range(len(content)):
        if type(content[i]) != str:
            content[i] = json.dumps(content[i])
    with open(file, "a") as f:
        writer = csv.writer(f)
        writer.writerow(content)


def evaluation(
    model,
    answer_data,
    prompt,
    output_data_path,
    skip_questions,
):
    for i in range(len(answer_data)):
        if answer_data[i].shape != answer_data[0].shape:
            raise ValueError("The number of rows in the answer files are different.")

    permutations = list(itertools.permutations(list(range(len(answer_data)))))

    if not os.path.exists(output_data_path):
        headers = ["request"]
        for i in range(len(permutations)):
            headers.append(f"{i}_{permutations[i]}_evaluation_request")
            headers.append(f"{i}_{permutations[i]}_evaluation")
            headers.append(f"{i}_{permutations[i]}_order")
        save(
            output_data_path,
            headers,
        )

    for i in tqdm(range(answer_data[0].shape[0])):
        row = []
        requests = [s.iloc[i]["request"] for s in answer_data]
        for request in requests:
            if request != requests[0]:
                raise ValueError(
                    f"The requests at line {i + 1} of the answer files are different."
                )
        if requests[0] in skip_questions:
            continue
        row.append(requests[0])
        for permutation in permutations:
            answers = [s.iloc[i]["response"] for s in answer_data]
            answers_reordered = [answers[j] for j in permutation]
            result = llm_zoo_evaluation(
                model=model,
                question=requests[0],
                answers=answers_reordered,
                prompt=prompt,
            )
            order_reorderd = [0] * len(result["order"])
            for j in range(len(result["order"])):
                order_reorderd[permutation[j]] = result["order"][j]
            row.append(result["evaluation_request"])
            row.append(result["response"])
            row.append(str(order_reorderd))
        save(
            output_data_path,
            row,
        )


if __name__ == "__main__":
    args, model = parse_args()

    if len(args.answer_file) <= 1:
        raise ValueError("At least two answer files are required.")

    output_data_path = os.path.join(args.output_folder, args.output_data_filename)

    skip_questions = set([])
    if os.path.exists(output_data_path):
        if args.yes or input(
            "Output file {} already exists. Override (Y/N)? ".format(output_data_path)
        ).lower() in {"yes", "y"}:
            print("Removing the existing output file...")
            os.remove(output_data_path)
        else:
            logging.info("Will skip existing results")
            skip_questions = pd.read_csv(output_data_path, usecols=["request"])
            skip_questions = set(skip_questions["request"].tolist())

    setup_logging(os.path.join(args.output_folder, args.log_filename))

    answer_data = []
    for answer_file in args.answer_file:
        answer_data.append(pd.read_csv(answer_file, usecols=["request", "response"]))

    evaluation(
        model=model,
        answer_data=answer_data,
        prompt=args.prompt,
        output_data_path=output_data_path,
        skip_questions=skip_questions,
    )
