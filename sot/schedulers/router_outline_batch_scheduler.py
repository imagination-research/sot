import re
import sys
import json
import logging
from collections import OrderedDict

import torch
from termcolor import colored
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .naive_scheduler import NaiveScheduler
from .outline_batch_scheduler import OutlineBatchScheduler
from sot.utils import _print_to_streams


class RouterOutlineBatchScheduler:
    def __init__(
        self,
        model,
        router_name_or_path,
        naive_prompt_file=None,
        outline_prompt_file=None,
        **kwargs,
    ):
        self._model = model
        self.router_tokenizer, self.router_model = self.load_router(router_name_or_path)
        self.naive_scheduler = NaiveScheduler(
            prompt_file=naive_prompt_file, model=self._model
        )
        self.outline_scheduler = OutlineBatchScheduler(
            prompt_file=outline_prompt_file, model=self._model
        )

    def load_router(self, router_name_or_path):
        model = AutoModelForSequenceClassification.from_pretrained(
            router_name_or_path,
            num_labels=2,
            local_files_only=True,
        ).cuda()
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(
            router_name_or_path,
            padding_size="right",
            use_fast=False,
            local_files_only=True,
        )
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer, model

    def get_fallback(self, request):
        input_ids = self.router_tokenizer(request, return_tensors="pt").input_ids.cuda()
        output = self.router_model(input_ids)
        return torch.argmax(output[0]).item()

    def set_model(self, model):
        self._model = model

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

        fallback = self.get_fallback(request[-1])

        if fallback == 0:
            return self.naive_scheduler.get_response(request, stream)
        else:
            return self.outline_scheduler.get_response(request, stream)
