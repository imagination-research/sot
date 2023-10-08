import argparse
import os
import csv
import json
from tqdm import tqdm
import pandas as pd
import logging

from models import get_model_class_from_name
from utils.logging import setup_logging
from evaluation.llm_evaluation import fastchat_evaluation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--answer-1-file", type=str, required=True)
    parser.add_argument("--answer-2-file", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument(
        "--output-data-filename", type=str, default="fastchat_evaluation.csv"
    )
    parser.add_argument(
        "--log-filename", type=str, default="fastchat_evaluation_log.log"
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
    answer1_data,
    answer2_data,
    template,
    prompt,
    output_data_path,
    skip_questions,
):
    if answer1_data.shape[0] != answer2_data.shape[0]:
        raise ValueError("The number of rows in the two answer files are different.")

    if not os.path.exists(output_data_path):
        save(
            output_data_path,
            [
                "request",
                "1_evaluation_request",
                "1_evaluation",
                "1_score1",
                "1_score2",
                "2_evaluation_request",
                "2_evaluation",
                "2_score1",
                "2_score2",
            ],
        )

    for i in tqdm(range(answer1_data.shape[0])):
        request1 = answer1_data.iloc[i]["request"]
        request2 = answer2_data.iloc[i]["request"]
        if request1 != request2:
            raise ValueError(
                f"The requests at line {i + 1} of the two answer files are different."
            )
        if request1 in skip_questions:
            continue
        answer1 = answer1_data.iloc[i]["response"]
        answer2 = answer2_data.iloc[i]["response"]
        result1 = fastchat_evaluation(
            model=model,
            template=template,
            question=request1,
            answer_1=answer1,
            answer_2=answer2,
            prompt=prompt,
        )
        result2 = fastchat_evaluation(
            model=model,
            template=template,
            question=request1,
            answer_1=answer2,
            answer_2=answer1,
            prompt=prompt,
        )
        save(
            output_data_path,
            [
                request1,
                result1["evaluation_request"],
                result1["response"],
                str(result1["score_pair"][0]),
                str(result1["score_pair"][1]),
                result2["evaluation_request"],
                result2["response"],
                str(result2["score_pair"][1]),
                str(result2["score_pair"][0]),
            ],
        )


if __name__ == "__main__":
    args, model = parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

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

    answer1_data = pd.read_csv(args.answer_1_file, usecols=["request", "response"])
    answer2_data = pd.read_csv(args.answer_2_file, usecols=["request", "response"])

    evaluation(
        model=model,
        answer1_data=answer1_data,
        answer2_data=answer2_data,
        template=args.template,
        prompt=args.prompt,
        output_data_path=output_data_path,
        skip_questions=skip_questions,
    )
