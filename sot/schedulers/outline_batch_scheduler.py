import re
import sys
import json
import copy
import logging
from collections import OrderedDict

from termcolor import colored

from .outline_scheduler import OutlineScheduler
from sot.utils import _print_to_streams


class OutlineBatchScheduler(OutlineScheduler):
    """
    OutlineBatchScheduler uses batch inference or the point-expanding stage.
    This class can be used for local models only.
    """

    def set_model(self, model):
        self._model = model

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
        raise NotImplementedError(
            "OutlineBatchScheduler currently doesn't implement file-based streaming, to"
            " see the streaming demo of OutlineBatchScheduler, please use the Gradio"
            " web demo in the repo."
        )

    def _get_response_stream(self, request):
        outline_request = request.copy()
        outline_ques, partial_answer = self.format_outline_prompt(request=request[-1])
        outline_request[-1] = outline_ques
        outline_request.append(partial_answer)
        for outputs in self._model.get_response([outline_request], stream=False):
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

        num_points = len(points)
        if num_points > 0:
            # Filter to get unique point indexes
            points_filtered = []
            point_outlines_filtered = []
            points_set = set([])
            for i in range(len(points)):
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

            for i_stream_out, outputs in enumerate(
                self._model.get_response(pe_requests, batch=True, stream=True)
            ):
                yield_outputs = copy.deepcopy(outputs)
                yield_outputs["stage"] = "expand"
                yield_outputs["ori_text"] = yield_outputs["text"]
                point_responses = [
                    point_resp.strip() for point_resp in yield_outputs["ori_text"]
                ]
                contents = [
                    partial_answer + " " + point_resp if partial_answer else point_resp
                    for (_, partial_answer), point_resp in zip(
                        pe_ques_and_partial_list, point_responses
                    )
                ]

                # Concatenate `contents` together as the new `outputs["text"]`
                # to show in the Gradio streaming demo
                yield_outputs["text"] = "\n".join(contents)
                # Note: When we need to change outputs["text"] based on outputs["text"],
                # we should deep copy the `outputs` dict instead of change it in place.
                # This can avoid second-time processing in the last loop (finish_reason=="stop"),
                # since the outputs["text"] will not be updated in the generation function.

                yield yield_outputs
            point_time = outputs["time"]
        else:
            contents = []

        yield {
            "stage": "summarize",
            "request": request,
            "response": "\n".join(contents),  # for main.py and prompt_eng_main.py
            "text": "\n".join(contents),  # for Gradio streaming demo
            "outline": outline,
            "outline_time": outline_time,
            "contents": contents,
            "points": points,
            "point_outlines": point_outlines,
            "point_time": point_time,
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
