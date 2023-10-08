import re
import sys
import json
import logging

from termcolor import colored

from .scheduler import Scheduler
from sot.utils import _print_to_streams


class OutlineScheduler(Scheduler):
    PROMPT_ROLE_SWITCH_STR = "[ROLESWITCHING assistant:]"

    def __init__(
        self, prompt_file=None, outline_prompt=None, point_prompt=None, **kwargs
    ):
        super().__init__(**kwargs)
        if prompt_file is not None:
            if outline_prompt is not None or point_prompt is not None:
                raise ValueError(
                    "When providing `prompt_file`, should not provide `outline_prompt`"
                    " and `point_prompt` through command-line arguments"
                )
            with open(prompt_file, "r") as rf:
                prompts = json.load(rf)
            self._outline_prompt = prompts["outline_prompt"]
            self._point_prompt = prompts["point_prompt"]
        else:
            if outline_prompt is None or point_prompt is None:
                raise ValueError(
                    "Should either provide `prompt_file`, or provide `outline_prompt`"
                    " and `point_prompt` through command-line arguments"
                )
            self._outline_prompt = outline_prompt
            self._point_prompt = point_prompt

    def print_info(self):
        super().print_info()
        logging.info(
            colored("OutlineScheduler *outline prompt*: ", "magenta")
            + f"'''{self._outline_prompt}'''"
        )
        logging.info(
            colored("OutlineScheduler *point prompt*: ", "magenta")
            + f"'''{self._point_prompt}'''"
        )

    @staticmethod
    def command_line_parser():
        parser = super(OutlineScheduler, OutlineScheduler).command_line_parser()
        parser.add_argument(
            "--prompt-file",
            type=str,
            help=(
                "The path of the JSON file containing `outline_prompt` and"
                " `point_prompt`."
            ),
            default=None,
        )
        parser.add_argument("--outline-prompt", type=str, default=None)
        parser.add_argument("--point-prompt", type=str, default=None)
        return parser

    def stream_output(self, output_generator, streams):
        if streams is None:
            streams = [sys.stderr]

        pre = 0
        output_text = ""
        cur_stage = "outline"
        logging.info(colored("Outline scheduler outline:", "magenta"))
        for outputs in output_generator:
            if outputs["stage"] == "summarize":
                _print_to_streams(streams, " ".join(output_text[pre:]), flush=True)
                _print_to_streams(streams, "\n\n", flush=True)
                return outputs
            if not outputs["stage"] == cur_stage:
                assert outputs["stage"] == "expand"
                assert outputs["point_index"] == 0
                _print_to_streams(streams, " ".join(output_text[pre:]), flush=True)
                _print_to_streams(streams, "\n\n", flush=True)
                logging.info(colored("Outline scheduler response:", "magenta"))
                cur_stage = outputs["stage"]
                pre = 0
                cur_point = 0
            if outputs["stage"] == "expand" and outputs["point_index"] != cur_point:
                _print_to_streams(streams, " ".join(output_text[pre:]), flush=True)
                _print_to_streams(streams, "\n\n", flush=True)
                pre = 0
                cur_point = outputs["point_index"]
            if "sub_request" in outputs:
                sub_request = outputs["sub_request"]
                logging.debug(
                    colored(f"Sub-request {cur_point}: '''{sub_request}'''", "magenta")
                )
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = (
                len(output_text) - 1
            )  # use len(output_text)-1 here since the last word might not finish
            if now > pre:
                _print_to_streams(
                    streams, " ".join(output_text[pre:now]), end=" ", flush=True
                )
                pre = now
        raise ValueError()

    def format_outline_prompt(self, request):
        splits = self._outline_prompt.split(self.PROMPT_ROLE_SWITCH_STR)
        if len(splits) == 1:
            return splits[0].format(request=request), None
        return splits[0].format(request=request), splits[1].format(request=request)

    def format_point_prompt(self, request, outline, point, point_outline):
        splits = self._point_prompt.split(self.PROMPT_ROLE_SWITCH_STR)
        if len(splits) == 1:
            return (
                splits[0].format(
                    request=request,
                    outline=outline,
                    point=point,
                    point_outline=point_outline,
                ),
                None,
            )
        return [
            split.format(
                request=request,
                outline=outline,
                point=point,
                point_outline=point_outline,
            )
            for split in splits
        ]

    def _get_response_stream(self, request, history=""):
        outline_request = request.copy()
        outline_ques, partial_answer = self.format_outline_prompt(request=request[-1])
        outline_request[-1] = outline_ques
        outline_request.append(partial_answer)
        logging.debug(colored(f"Outline request: {outline_request}\n----", "magenta"))
        for outputs in self._model.get_response([outline_request], stream=True):
            outputs["stage"] = "outline"
            yield outputs
        outline = outputs["text"]
        outline_time = outputs["time"]
        if partial_answer:
            outline = partial_answer + outline

        # Extract points.
        re_result = re.findall(r"(\d+)\.\s?([\s\S]+?)(?=\n|\n*$)", outline)
        if len(re_result) > 0:
            points, point_outlines = zip(*re_result)
        else:
            points, point_outlines = [], []
        assert len(points) == len(point_outlines)

        num_points = len(points)
        contents_time = []
        if num_points > 0:
            # Filter to get unique point indexes
            points_filtered = []
            point_outlines_filtered = []
            points_set = set([])
            for i in range(num_points):
                if points[i] not in points_set:
                    points_set.add(points[i])
                    points_filtered.append(points[i])
                    point_outlines_filtered.append(point_outlines[i])
            points = points_filtered
            point_outlines = point_outlines_filtered

            pe_ques_and_partial_list = [
                self.format_point_prompt(
                    request=request[-1],
                    point=point,
                    outline=outline,
                    point_outline=point_outline,
                )
                for point, point_outline in zip(points, point_outlines)
            ]
            pe_requests = [request.copy() for _ in range(len(points))]
            for pe_request, (pe_ques, pe_partial) in zip(
                pe_requests, pe_ques_and_partial_list
            ):
                pe_request[-1] = pe_ques
                pe_request.append(pe_partial)

            contents = []
            for i_point, sub_request in enumerate(pe_requests):
                for i_stream_out, outputs in enumerate(
                    self._model.get_response([sub_request], stream=True)
                ):
                    if i_stream_out == 0:
                        outputs["sub_request"] = sub_request
                    outputs["stage"] = "expand"
                    outputs["point_index"] = i_point
                    yield outputs
                # append the text of the point #i_point to `contents`
                contents.append(outputs["text"])
                contents_time.append(outputs["time"])
            # for final response, concatenate the partial answer together
            for i_point, (_, partial_answer) in enumerate(pe_ques_and_partial_list):
                if partial_answer:
                    contents[i_point] = partial_answer + " " + contents[i_point]
        else:
            contents = []

        yield {
            "stage": "summarize",
            "request": request,
            "response": "\n\n".join(contents),
            "outline": outline,
            "contents": contents,
            "points": points,
            "point_outlines": point_outlines,
            "outline_time": outline_time,
            "contents_time": contents_time,
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
