import openai
import os
from tqdm import tqdm
import logging
import time
from utils.logging import logger
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)
from itertools import chain

from .model import Model


class OpenAIModel(Model):
    def __init__(
        self,
        api_type,
        api_base,
        api_version,
        temperature,
        max_tokens,
        top_p,
        frequency_penalty,
        presence_penalty,
        engine,
        api_model,
        system_message,
        timeout,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._api_type = api_type
        self._api_base = api_base
        self._api_version = api_version
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty
        self._engine = engine
        self._api_model = api_model
        self._system_message = system_message
        self._timeout = timeout

        openai.api_type = self._api_type
        openai.api_base = self._api_base
        if self._api_version is not None:
            openai.api_version = self._api_version
        openai.api_key = os.getenv("OPENAI_API_KEY")

    @staticmethod
    def command_line_parser():
        parser = super(OpenAIModel, OpenAIModel).command_line_parser()
        parser.add_argument("--api-type", type=str, required=True)
        parser.add_argument("--api-base", type=str)
        parser.add_argument("--api-version", type=str, default=None)
        parser.add_argument("--temperature", type=float)
        parser.add_argument("--max-tokens", type=int)
        parser.add_argument("--top-p", type=float)
        parser.add_argument("--frequency-penalty", type=float)
        parser.add_argument("--presence-penalty", type=float)
        parser.add_argument("--engine", type=str, default=None)
        parser.add_argument("--api-model", type=str, default=None)
        parser.add_argument("--system-message", type=str)
        parser.add_argument("--timeout", type=float, default=60)
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
        retry=retry_if_not_exception_type(
            (
                openai.error.InvalidRequestError,
                openai.error.AuthenticationError,
            )
        ),
        wait=wait_random_exponential(min=8, max=500),
        stop=stop_after_attempt(30),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
    )
    def _get_reponse_for_one_request(self, request):
        if isinstance(request, str):
            request = [request]
        messages = [
            {
                "role": "system",
                "content": self._system_message,
            }
        ]
        roles = ["user", "assistant"]
        for r_i, r in enumerate(request):
            if r is not None:
                messages.append({"role": roles[r_i % 2], "content": r})
        start = time.time()
        response = openai.ChatCompletion.create(
            engine=self._engine,
            model=self._api_model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            top_p=self._top_p,
            frequency_penalty=self._frequency_penalty,
            presence_penalty=self._presence_penalty,
            stop=None,
            request_timeout=self._timeout,
        )
        end = time.time()
        response = response["choices"][0]["message"]["content"]
        elapsed_time = end - start
        return {"text": response, "time": elapsed_time}
