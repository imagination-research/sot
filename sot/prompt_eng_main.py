import os
import re
import sys
import csv
import yaml
import json
import shutil
import logging
import argparse
import readline
from collections import OrderedDict

from tqdm import tqdm
import pandas as pd
import IPython
from termcolor import colored

from models import get_model_class_from_name
from schedulers import get_scheduler_class_from_name
from utils.logging import setup_logging

# ---- Copy from https://stackoverflow.com/questions/5637124/tab-completion-in-pythons-raw-input ----
COMMANDS = [
    "usedata",
    "useprompt",
    "test",
    "exit",
    "embedipy",
    "setloglevel",
    "usenaiveprompt",
]
COMMAND_HINTS = ["<filename>", "<scheduler_type> <data_index>", "", ""]
RE_SPACE = re.compile(".*\s+$", re.M)


class Completer(object):
    def _listdir(self, root):
        "List directory 'root' appending the path separator to subdirs."
        res = []
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                name += os.sep
            res.append(name)
        return res

    def _complete_path(self, path=None):
        "Perform completion of filesystem path."
        if not path:
            return self._listdir(".")
        dirname, rest = os.path.split(path)
        tmp = dirname if dirname else "."
        res = [
            os.path.join(dirname, p) for p in self._listdir(tmp) if p.startswith(rest)
        ]
        # more than one match, or single match which does not exist (typo)
        if len(res) > 1 or not os.path.exists(path):
            return res
        # resolved to a single directory, so return list of files below it
        if os.path.isdir(path):
            return [os.path.join(path, p) for p in self._listdir(path)]
        # exact file match terminates this completion
        return [path + " "]

    def complete_useprompt(self, args):
        if not args:
            return self._complete_path(".")
        # treat the last arg as a path and complete it
        return self._complete_path(args[-1])

    def complete_usedata(self, args):
        if not args:
            return self._complete_path(".")
        # treat the last arg as a path and complete it
        return self._complete_path(args[-1])

    def complete_usenaiveprompt(self, args):
        if not args:
            return self._complete_path(".")
        # treat the last arg as a path and complete it
        return self._complete_path(args[-1])

    def complete(self, text, state):
        "Generic readline completion entry point."
        buffer = readline.get_line_buffer()
        line = readline.get_line_buffer().split()
        # show all commands
        if not line:
            return [c + " " + c_hint for c, c_hint in zip(COMMANDS, COMMAND_HINTS)][
                state
            ]
        # account for last argument ending in a space
        if RE_SPACE.match(buffer):
            line.append("")
        # resolve command to the implementation function
        cmd = line[0].strip()
        if cmd in COMMANDS:
            impl = getattr(self, "complete_%s" % cmd)
            args = line[1:]
            if args:
                return (impl(args) + [None])[state]
            return [cmd + " "][state]
        results = [c + " " for c in COMMANDS if c.startswith(cmd)] + [None]
        return results[state]


comp = Completer()
# we want to treat '/' as part of a word, so override the delimiters
readline.set_completer_delims(" \t\n;")
readline.parse_and_bind("tab: complete")
readline.set_completer(comp.complete)
# ---- End copy from https://stackoverflow.com/questions/5637124/tab-completion-in-pythons-raw-input ----


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-log", type=str, required=True)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Streaming the model outputs to the console.",
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
    # model, other_args = None, [] # for debug

    if other_args != []:
        raise ValueError("Unknown arguments: {}".format(other_args))

    return args, model


def save(file, content):
    for i in range(len(content)):
        if type(content[i]) != str:
            content[i] = json.dumps(content[i])
    with open(file, "a", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(content)


def useprompt(cmd, model, input_data, schedulers):
    if not len(cmd) == 2:
        logging.error("useprompt takes 1 argument")
        return
    try:
        schedulers["outline"] = get_scheduler_class_from_name("outline")(
            model=model, prompt_file=cmd[1]
        )
    except Exception as error:
        logging.error(f"{error.__class__.__name__}: {error}")
        return
    schedulers["outline"].print_info()


def usenaiveprompt(cmd, model, input_data, schedulers):
    if not len(cmd) == 2:
        logging.error("usenaiveprompt takes 1 argument")
        return
    if cmd[1] == "none":
        # clear the naive prompt template
        schedulers["naive"] = get_scheduler_class_from_name("naive")(model=model)
    else:
        try:
            schedulers["naive"] = get_scheduler_class_from_name("naive")(
                model=model, prompt_file=cmd[1]
            )
        except Exception as error:
            logging.error(f"{error.__class__.__name__}: {error}")
            return
    schedulers["naive"].print_info()


def usedata(cmd, model, data_container, schedulers):
    if not len(cmd) == 2:
        logging.error("usedata takes 1 argument")
        return
    try:
        data_container["data"] = pd.read_csv(cmd[1], header=0, names=["request"])
        logging.info(
            "Loaded data from %s, %d questions in total",
            cmd[1],
            len(data_container["data"]),
        )
    except Exception as error:
        logging.error(f"{error.__class__.__name__}: {error}")
        return


def test(cmd, model, input_data, schedulers, log_filename):
    if not len(cmd) in {2, 3}:
        logging.error(
            "test takes 1 or 2 argument: 'test [optional <scheduler_type>,]"
            " <data_index>"
        )
        return
    if len(cmd) == 2:
        scheduler_type = "outline"
        data_index = cmd[1]
    else:
        scheduler_type = cmd[1]
        data_index = cmd[2]
    try:
        data_index = int(data_index)
    except ValueError as e:
        logging.error(e)
        return

    if scheduler_type not in schedulers:
        logging.error(
            f"Scheduler {scheduler_type} not exist. Available: {schedulers.keys()}"
        )
        return
    if data_index >= input_data.shape[0]:
        logging.error(f"Data index {data_index} not exist. #rows={input_data.shape[0]}")
        return

    # test
    scheduler = schedulers[scheduler_type]
    request = input_data.iloc[data_index]["request"]
    logging.info("Testing requeset: %s", request)
    try:
        if args.stream:
            if scheduler_type == "batch_outline":
                logging.warning(
                    "`batch_outline` scheduler doesn't support streaming to the"
                    " console, let us use the non-streaming output mode."
                )
                response = scheduler.get_response(request)
            else:
                # For rapid iterating the prompts, let's change into streaming
                # (i.e., as long as we identify some problem,
                # we can 1. abort the genertion, 2. change the prompt template, 3. retest.
                output_generator = scheduler.get_response(request, stream=True)
                with open(log_filename, "a", encoding="utf-8") as logfile_stream:
                    response = scheduler.stream_output(
                        output_generator, streams=[sys.stderr, logfile_stream]
                    )
        else:
            response = scheduler.get_response(request)
    except KeyboardInterrupt:
        logging.info("Previous generation interrupted.")
        return

    # print the stats, and the responses (if not streaming)
    if scheduler_type == "naive":
        if not args.stream:
            logging.info("Naive scheduler response: %s", response["response"])

        stats = OrderedDict(
            num_history_round=(len(response["request"]) - 1) // 2,
            request_length=len(response["request"][-1]),
            response_length=len(response["response"]),
            request_tokens=len(model.tokenizer(response["request"][-1])["input_ids"]),
            response_tokens=len(model.tokenizer(response["response"])["input_ids"]),
        )
        logging.info(
            colored(
                "Naive scheduler stats: {}".format(
                    ", ".join([f"{key}={value}" for key, value in stats.items()])
                ),
                "green",
            )
        )
    else:  # if scheduler_type == "outline"
        if not args.stream or scheduler_type == "batch_outline":
            # not streaming, print out the response first
            logging.info(f"Outline scheduler outline: %s", response["outline"])
            logging.info("Outline scheduler response: %s", response["response"])

        stats = OrderedDict(
            num_history_round=(len(response["request"]) - 1) // 2,
            request_length=len(response["request"][-1]),
            outline_length=len(response["outline"]),
            num_points=len(response["points"]),
            point_outline_length=[len(point) for point in response["point_outlines"]],
            point_response_length=[len(content) for content in response["contents"]],
            response_length=len(response["response"]),
            request_tokens=len(model.tokenizer(response["request"][-1])["input_ids"]),
            outline_tokens=len(model.tokenizer(response["outline"])["input_ids"]),
            point_outline_tokens=[
                len(model.tokenizer(point)["input_ids"])
                for point in response["point_outlines"]
            ],
            point_response_tokens=[
                len(model.tokenizer(content)["input_ids"])
                for content in response["contents"]
            ],
            response_tokens=len(model.tokenizer(response["response"])["input_ids"]),
        )
        logging.info(
            colored(
                "Outline scheduler stats: {}".format(
                    ", ".join([f"{key}={value}" for key, value in stats.items()])
                ),
                "green",
            )
        )
    return response


DEFAULT_DATA_PATH = "data/vicuna/data.csv"
DEFAULT_PROMPT_PATH = "prompts/sot_opensource.json"

if __name__ == "__main__":
    args, model = parse_args()

    dir_name = os.path.dirname(args.output_log)
    os.makedirs(dir_name, exist_ok=True)

    setup_logging(args.output_log)

    data_container = {
        "data": pd.read_csv(DEFAULT_DATA_PATH, header=0, names=["request"])
    }

    schedulers = {
        "outline": get_scheduler_class_from_name("outline")(
            model=model, prompt_file=DEFAULT_PROMPT_PATH
        ),
        "batch_outline": get_scheduler_class_from_name("batch_outline")(
            model=model, prompt_file=DEFAULT_PROMPT_PATH
        ),
        "naive": get_scheduler_class_from_name("naive")(model=model),
    }
    schedulers["outline"].print_info()

    while 1:
        try:
            cmd = input(
                colored(
                    f"Please give a CMD (supported commands: {COMMANDS}): ", "green"
                )
            )
            # 'useprompt <filename>' or 'test <scheduler_type <data_index>'"
            cmd = re.split("\s+", cmd.strip())
            if cmd[0] not in COMMANDS:
                logging.error(f"Only these commands are supported: {COMMANDS}")
                continue
            if cmd[0] == "useprompt":
                useprompt(cmd, model, data_container["data"], schedulers)
            elif cmd[0] == "usenaiveprompt":
                usenaiveprompt(cmd, model, data_container["data"], schedulers)
            elif cmd[0] == "test":
                response = test(
                    cmd, model, data_container["data"], schedulers, args.output_log
                )
            elif cmd[0] == "usedata":
                usedata(cmd, model, data_container, schedulers)
            elif cmd[0] == "embedipy":
                IPython.embed()
            elif cmd[0] == "setloglevel":
                if len(cmd) != 2:
                    logging.error(
                        "`setloglevel` accept one and only one log-level argument."
                    )
                    continue
                level = getattr(logging, cmd[1].upper(), None)
                if level is None:
                    logging.error(f"Level '{cmd[1]}' not exists")
                    continue
                logging.getLogger().setLevel(level)
                logging.info(f"Set log level to {level} ({cmd[1]})")
            elif cmd[0] == "exit":
                logging.info("Exit.")
                sys.exit(0)
        except KeyboardInterrupt:
            logging.error(f"Only these commands are supported: {COMMANDS}")
            continue
