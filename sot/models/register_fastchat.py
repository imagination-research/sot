from transformers import LlamaTokenizer, LlamaForCausalLM
from fastchat.model.model_adapter import BaseModelAdapter, register_model_adapter
from fastchat.conversation import (
    register_conv_template,
    get_conv_template,
    SeparatorStyle,
    Conversation,
)


## ---- Register model adapters ----
class OrcaLLaMAAdapter(BaseModelAdapter):
    def match(self, model_path: str):
        return "orca" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path, **from_pretrained_kwargs)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("orca")


class OpenChatLLaMAAdapter(BaseModelAdapter):
    def match(self, model_path: str):
        return "openchat" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path, **from_pretrained_kwargs)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("openchat")


"""
# Note: Due to FastChat's matching mechanism and matching criterion for Vicuna,
#       we must register it before registering VicunaAdapter for StableVicuna to
#       successfully match this StableVicunaAdapter
class StableVicunaAdapter(BaseModelAdapter):
    def match(self, model_path: str):
        return "stable-vicuna" in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("stable_vicuna")
"""


class UltraLMAdapter(BaseModelAdapter):
    def match(self, model_path: str):
        return "ultralm" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("ultra-lm")


register_model_adapter(OrcaLLaMAAdapter)
register_model_adapter(OpenChatLLaMAAdapter)
register_model_adapter(UltraLMAdapter)

## ---- Register conversation templates ----

# ref: https://huggingface.co/psmathur/orca_mini_13b
register_conv_template(
    Conversation(
        name="orca",
        system_message=(
            "### System:\nYou are an AI assistant that follows instruction extremely"
            " well. Help as much as you can.\n\n"
        ),
        roles=("### User:\n", "### Response:\n"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n\n",
    )
)


# ref: https://huggingface.co/openchat/openchat
#      https://github.com/imoneoi/openchat/blob/master/ochat/serving/inference.py
register_conv_template(
    Conversation(
        name="openchat",
        system_message=(
            "System: You are an AI assistant that follows instruction extremely well."
            " Help as much as you can.\n"
        ),
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="<|end_of_turn|>",
        stop_token_ids=[32000],
    )
)

"""
# ref: https://huggingface.co/CarperAI/stable-vicuna-13b-delta
register_conv_template(
    Conversation(
        name="stable_vicuna",
        system_message="",
        roles=("### Human", "### Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="###",
    )
)
"""


# ref: https://huggingface.co/openbmb/UltraLM-13b
register_conv_template(
    Conversation(
        name="ultra-lm",
        system_message=(
            "A chat between a curious user and an artificial intelligence assistant."
            " The assistant gives helpful, detailed, and polite answers to the user's"
            " questions."
        ),
        roles=("User", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="</s>",
    )
)
