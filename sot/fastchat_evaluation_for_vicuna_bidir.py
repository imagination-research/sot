import argparse
import os
import csv
import json
from tqdm import tqdm
import pandas as pd
import requests
import tempfile
import logging

from models import get_model_class_from_name
from utils.logging import setup_logging
from evaluation.llm_evaluation import fastchat_evaluation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--question-category-url",
        type=str,
        default=(
            "https://raw.githubusercontent.com/lm-sys/FastChat/640ec6205031955e841523e"
            "1a8606afb4d0538c2/fastchat/eval/table/question.jsonl"
        ),
    )
    parser.add_argument(
        "--prompt-url",
        type=str,
        default=(
            "https://raw.githubusercontent.com/lm-sys/FastChat/640ec6205031955e841523e"
            "1a8606afb4d0538c2/fastchat/eval/table/prompt.jsonl"
        ),
    )
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
    output_data_path,
    question_to_prompt,
    question_to_template,
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
        if request1 not in question_to_template or request1 not in question_to_prompt:
            raise ValueError(
                f"The request {request1} at line {i + 1} of the two answer files "
                "is not in the prompt file."
            )
        result1 = fastchat_evaluation(
            model=model,
            template=question_to_template[request1],
            question=request1,
            answer_1=answer1,
            answer_2=answer2,
            prompt=question_to_prompt[request1],
        )
        result2 = fastchat_evaluation(
            model=model,
            template=question_to_template[request1],
            question=request1,
            answer_1=answer2,
            answer_2=answer1,
            prompt=question_to_prompt[request1],
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


def download(url, file):
    response = requests.get(url)
    with open(file, "wb") as f:
        f.write(response.content)


def download_template(question_category_url, prompt_url):
    with tempfile.TemporaryDirectory() as tmp_dir:
        question_category_file = os.path.join(tmp_dir, "question.jsonl")
        prompt_file = os.path.join(tmp_dir, "prompt.jsonl")
        download(question_category_url, question_category_file)
        download(prompt_url, prompt_file)
        question_to_category = {}
        with open(question_category_file, "r") as f:
            for line in f:
                content = line.strip()
                content = json.loads(content)
                question = content["text"]
                category = content["category"]
                assert question not in question_to_category
                question_to_category[question] = category
        category_to_prompt = {}
        category_to_template = {}
        with open(prompt_file, "r") as f:
            for line in f:
                content = line.strip()
                content = json.loads(content)
                template = content["prompt_template"]
                prompt = content["defaults"]["prompt"]
                category = content["category"]
                assert category not in category_to_prompt
                assert category not in category_to_template
                category_to_prompt[category] = prompt
                category_to_template[category] = template
    question_to_prompt = {}
    question_to_template = {}
    for question, category in question_to_category.items():
        if category not in category_to_prompt:
            category = "general"
        question_to_prompt[question] = category_to_prompt[category]
        question_to_template[question] = category_to_template[category]
    return question_to_prompt, question_to_template


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

    question_to_prompt, question_to_template = download_template(
        args.question_category_url, args.prompt_url
    )

    evaluation(
        model=model,
        answer1_data=answer1_data,
        answer2_data=answer2_data,
        output_data_path=output_data_path,
        question_to_prompt=question_to_prompt,
        question_to_template=question_to_template,
        skip_questions=skip_questions,
    )
