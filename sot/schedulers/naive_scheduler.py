import sys
import json
import logging

from termcolor import colored

from .scheduler import Scheduler
from sot.utils import _print_to_streams


class NaiveScheduler(Scheduler):
    def __init__(self, prompt_file=None, **kwargs):
        super().__init__(**kwargs)
        if prompt_file is not None and prompt_file != "none":
            with open(prompt_file, "r") as rf:
                prompts = json.load(rf)
            self.prompt = prompts["prompt"]
        else:
            self.prompt = "{request}"

    def set_model(self, model):
        self._model = model

    def print_info(self):
        super().print_info()
        logging.info(
            colored("NaiveScheduler *prompt*: ", "magenta") + f"'''{self.prompt}'''"
        )

    @staticmethod
    def command_line_parser():
        parser = super(NaiveScheduler, NaiveScheduler).command_line_parser()
        parser.add_argument(
            "--prompt-file",
            type=str,
            help=(
                "The path of the JSON file containing `prompt`. "
                "'--promptfile none' is equivalent to not specifying this argument."
            ),
            default=None,
        )
        return parser

    def stream_output(self, output_stream, streams=None):
        if streams is None:
            streams = [sys.stderr]
        pre = 0
        for outputs in output_stream:
            if outputs.get("stage", None) == "summarize":
                _print_to_streams(streams, " ".join(output_text[pre:]), flush=True)
                _print_to_streams(streams, "\n\n", flush=True)
                return outputs
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                _print_to_streams(
                    streams, " ".join(output_text[pre:now]), end=" ", flush=True
                )
                pre = now
        raise ValueError()

    def format_outline_prompt(self, request):
        return self.prompt.format(request=request)

    def _get_response_stream(self, request):
        request = request.copy()
        ques = self.format_outline_prompt(request[-1])
        request[-1] = ques
        for outputs in self._model.get_response([request], stream=True):
            yield outputs

        yield {
            "stage": "summarize",
            "request": request,
            "text": outputs["text"],
            "response": outputs["text"],
            "time": outputs["time"],
        }

    def get_response(self, request, stream=False):
        if isinstance(request, str):
            # one request should be a list of messages,
            # alternatively from the user and the assistant
            request = [request]
        if len(request) % 2 != 1:
            raise ValueError(
                "The length of the request messages should be odd."
                "So that the final message is from the user."
            )

        if stream:
            return self._get_response_stream(request)

        for outputs in self._get_response_stream(request):
            pass
        return outputs
