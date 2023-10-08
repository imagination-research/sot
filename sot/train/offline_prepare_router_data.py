import pickle
from typing import Dict
from dataclasses import dataclass, field

import pandas as pd
import torch
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="roberta-base")
    model_max_length: int = field(default=512)


@dataclass
class DataArguments:
    data_path: str = field(
        metadata={"help": "A csv file containing the training data."}
    )
    output_data_path: str = field(
        metadata={"help": "The pickle file name to dump the training data."}
    )


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    questions = list(sources["request"])
    labels = torch.tensor(sources["label"]).to(torch.long)

    inputs = tokenizer(
        questions,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    assert len(input_ids) == len(labels)

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # load the raw data
    raw_data = pd.read_csv(data_args.data_path, delimiter=";")

    # preprocess
    data = preprocess(raw_data, tokenizer)

    print("Data shape:", data["input_ids"].shape)

    with open(data_args.output_data_path, "wb") as wf:
        print(f"Pickle dumping the data to {data_args.output_data_path}")
        pickle.dump(data, wf)
