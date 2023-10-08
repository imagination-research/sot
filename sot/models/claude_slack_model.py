import os
from tqdm import tqdm
from utils.logging import logger
import logging
import time
import errno
import signal
import functools
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from itertools import chain

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)

from .model import Model


# https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


class ClaudeSlackModel(Model):
    def __init__(
        self,
        channel_name,
        first_message_timeout,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._slack_user_token = os.getenv("SLACK_USER_TOKEN")
        self._channel_name = channel_name
        self._first_message_timeout = first_message_timeout

        self._client = WebClient(token=self._slack_user_token)

        # Get claude bot user id.
        self._bot_user_id = None
        response = self._client.users_list()
        for member in response["members"]:
            if member["name"] == "claude":
                self._bot_user_id = member["id"]
                break
        if self._bot_user_id is None:
            raise ValueError("Claude bot user id not found.")

        try:
            response = self._client.conversations_create(name=self._channel_name)
            self._channel_id = response["channel"]["id"]
            self._client.conversations_invite(
                channel=self._channel_id, users=self._bot_user_id
            )
        except SlackApiError as e:
            logging.error(f"Claude error: {e}")
            self._channel_id = None
            response = self._client.conversations_list()
            for i in range(len(response["channels"])):
                if response["channels"][i]["name"] == self._channel_name:
                    self._channel_id = response["channels"][i]["id"]
                    break
            if self._channel_id is None:
                raise ValueError(
                    f"Channel {self._channel_name} does not exist in Slack."
                )

    @staticmethod
    def command_line_parser():
        parser = super(ClaudeSlackModel, ClaudeSlackModel).command_line_parser()
        parser.add_argument("--channel_name", type=str, required=True)
        parser.add_argument("--first_message_timeout", type=float, default=10)
        return parser

    def get_response(self, requests, stream=False):
        if stream:
            return chain(
                *[
                    self._get_response_for_one_request_stream(request)
                    for request in requests
                ]
            )
        else:
            return [
                self._get_reponse_for_one_request(request) for request in tqdm(requests)
            ]

    def _get_response_for_one_request_stream(self, request):
        # TODO: implement streaming mode
        response = self._get_reponse_for_one_request(request)
        yield response

    @retry(
        wait=wait_random_exponential(min=10, max=500),
        stop=stop_after_attempt(30),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
    )
    @timeout(60)
    def _get_reponse_for_one_request(self, request):
        if isinstance(request, str):
            request = [request]
        if isinstance(request, (tuple, list)):
            request = [r for r in request if r is not None]
        request = " ".join(request)
        request = f"<@{self._bot_user_id}> " + request
        start = time.time()
        response = self._client.chat_postMessage(channel=self._channel_id, text=request)
        timestamp = response["ts"]
        response_text = None
        while response_text is None:
            response = self._client.conversations_replies(
                channel=self._channel_id, ts=timestamp
            )
            if len(response["messages"]) > 1:
                response_text = response["messages"][1]["text"]
                if response_text.endswith("Typing…_"):
                    response_text = None
            else:
                if time.time() - start > self._first_message_timeout:
                    raise ValueError("First message timeout.")
            time.sleep(1)
        end = time.time()
        elapsed_time = end - start
        return {"text": response_text, "time": elapsed_time}
