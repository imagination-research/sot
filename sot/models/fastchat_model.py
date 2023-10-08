from itertools import chain
import torch

from fastchat.model.model_adapter import (
    add_model_args,
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.conversation import get_conv_template
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.awq import AWQConfig
from fastchat.utils import get_context_length

from .model import Model
from .batch_inference import batch_generate_stream
from . import register_fastchat


class FastChatModel(Model):
    def __init__(
        self,
        model_path,
        device,
        gpus,
        num_gpus,
        max_gpu_memory,
        load_8bit,
        cpu_offloading,
        gptq_ckpt,
        gptq_wbits,
        gptq_groupsize,
        gptq_act_order,
        awq_ckpt,
        awq_wbits,
        awq_groupsize,
        conv_template,
        temperature,
        repetition_penalty,
        max_new_tokens,
        revision,
        **kwargs,
    ):
        super().__init__()
        self._model_path = model_path
        self._device = device
        self._num_gpus = num_gpus
        self._max_gpu_memory = max_gpu_memory
        self._load_8bit = load_8bit
        self._cpu_offloading = cpu_offloading
        self._gptq_config = GptqConfig(
            ckpt=gptq_ckpt or self._model_path,
            wbits=gptq_wbits,
            groupsize=gptq_groupsize,
            act_order=gptq_act_order,
        )
        self._awq_config = AWQConfig(
            ckpt=awq_ckpt or self._model_path,
            wbits=awq_wbits,
            groupsize=awq_groupsize,
        )
        self._conv_template = conv_template
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty
        self._max_new_tokens = max_new_tokens
        self._revision = revision

        self.model, self.tokenizer = load_model(
            model_path=self._model_path,
            device=self._device,
            num_gpus=self._num_gpus,
            max_gpu_memory=self._max_gpu_memory,
            load_8bit=self._load_8bit,
            cpu_offloading=self._cpu_offloading,
            gptq_config=self._gptq_config,
            awq_config=self._awq_config,
            revision=self._revision,
            # **kwargs,
        )

        # use padding or EOS to do *left padding* for batched point-expanding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_size = "left"

        # streaming generation func
        self.generate_stream_func = get_generate_stream_function(
            self.model, self._model_path
        )
        # batched streaming generation func
        self.generate_batch_stream_func = batch_generate_stream

        self.context_len = get_context_length(self.model.config)

    @staticmethod
    def command_line_parser():
        parser = super(FastChatModel, FastChatModel).command_line_parser()
        add_model_args(parser)
        parser.add_argument(
            "--conv-template",
            type=str,
            default=None,
            help="Conversation prompt template.",
        )
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--repetition_penalty", type=float, default=1.0)
        parser.add_argument("--max-new-tokens", type=int, default=512)
        return parser

    def set_params(self, temperature, repetition_penalty, max_new_tokens):
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty
        self._max_new_tokens = max_new_tokens

    def get_response(self, requests, batch=False, stream=False):
        if stream:
            if not batch:
                # return the generator that is the sequential chain
                # of multiple generators, each handling one request
                return chain(
                    *[
                        self._get_response_for_one_request(
                            request, batch=False, stream=True
                        )
                        for request in requests
                    ]
                )
            else:
                # return the generator, in which multiple requests
                # will be handled by batch inference
                return self._get_response_for_one_request(
                    requests, batch=True, stream=True
                )
        if not batch:
            return [
                self._get_response_for_one_request(request, batch=False)
                for request in requests
            ]
        else:
            return self._get_response_for_one_request(requests, batch=True)

    def _get_response_for_one_request(self, request, batch=False, stream=False):
        if stream:
            # streaming mode: return the generator
            return self._get_response_for_one_request_stream(request, batch=batch)

        # non-streaming mode: drain the generator and return
        for outputs in self._get_response_for_one_request_stream(request, batch=batch):
            pass
        return outputs

    def _get_response_for_one_request_stream(self, request, batch=False):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        starter.record()

        if batch:
            # request is a list of request, each containing multiple messages
            # handle multiple requests with batched inference
            results = [self._get_prompt(single_req) for single_req in request]
            stop_str, stop_token_ids = results[0][1:]
            prompt = [res[0] for res in results]
            generate_stream_func = self.generate_batch_stream_func
        else:
            # request is a single request containing multiple messages
            # handle single request
            prompt, stop_str, stop_token_ids = self._get_prompt(request)
            generate_stream_func = self.generate_stream_func

        gen_params = {
            "model": self._model_path,
            "prompt": prompt,
            "temperature": self._temperature,
            "repetition_penalty": self._repetition_penalty,
            "max_new_tokens": self._max_new_tokens,
            "stop": stop_str,
            "stop_token_ids": stop_token_ids,
            "echo": False,
        }

        output_stream = generate_stream_func(
            self.model,
            self.tokenizer,
            gen_params,
            self._device,
            context_len=self.context_len,
        )

        for outputs in output_stream:
            yield outputs

        ender.record()
        torch.cuda.synchronize()
        elapsed_time = starter.elapsed_time(ender)
        outputs["time"] = elapsed_time / 1000
        yield outputs

    def _get_prompt(self, request):
        if self._conv_template:
            conv = get_conv_template(self._conv_template)
        else:
            conv = get_conversation_template(self._model_path)

        # clear the template messages, and sometimes including the system prompt
        conv.messages = []
        num_history_round = (len(request) - 1) // 2
        # add the history messages
        for i_message, message in enumerate(request[: num_history_round * 2]):
            conv.append_message(conv.roles[i_message % 2], message)
        # add the user question at this round
        conv.append_message(conv.roles[0], request[num_history_round * 2])
        # indicate it's the assistant's turn to answer
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        if len(request) == num_history_round * 2 + 2:
            # have partial answer for the assistant, add the partial answer to the prompt
            partial_answer = request[num_history_round * 2 + 1] or ""
            prompt += partial_answer
        return prompt, conv.stop_str, conv.stop_token_ids
