# This code is based on lm-sys/FastChat and tatsu-lab/stanford_alpaca. Below is the original copyright
# from tatsu-label/stanford_alpaca.
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import csv
import pickle
import pathlib
from typing import Dict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import evaluate
import transformers
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification


SEED = 0


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="roberta-base")
    cache_dir: str = field(default=None)
    num_labels: int = field(default=2)
    model_max_length: int = field(default=512)


@dataclass
class DataArguments:
    data_path: str = field(
        default="lima_router.pkl",
        metadata={"help": "A pickle file containing the processed LIMA data."},
    )
    vicuna_data_path: str = field(default="vicuna_router.pkl")
    wizardlm_data_path: str = field(default="wizardlm_router.pkl")
    vicuna_pred_path: str = field(default="vicuna_router_pred.csv")
    wizardlm_pred_path: str = field(default="wizardlm_router_pred.csv")
    train_val_ratio: float = field(default=0.8)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default=".")
    num_train_epochs: int = field(default=2)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=32)
    optim: str = field(default="adamw_torch")
    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.01)
    label_smoothing: float = field(default=0.9)
    tversky_ratio: float = field(default=0.75)
    fp_ratio: float = field(default=0.7)
    fn_ratio: float = field(default=0.3)
    evaluation_strategy: str = field(default="epoch")


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_dict):
        super(SupervisedDataset, self).__init__()

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


def make_supervised_data_module(
    data_dict: Dict[str, torch.Tensor], train_val_ratio: float
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset

    # Split train/test
    np.random.seed(SEED)
    perm = np.random.permutation(data_dict["input_ids"].shape[0])
    split = int(len(perm) * train_val_ratio)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_data_dict = {key: tensor[train_indices] for key, tensor in data_dict.items()}
    eval_data_dict = {key: tensor[eval_indices] for key, tensor in data_dict.items()}
    print(f"#train {len(train_indices)}, #eval {len(eval_indices)}")

    train_dataset = dataset_cls(train_data_dict)
    eval_dataset = dataset_cls(eval_data_dict)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def load_dataset(data_path):
    with open(data_path, "rb") as rf:
        data_dict = pickle.load(rf)
    return data_dict


def save_predictions(results, pred_path):
    predictions = np.argmax(results.predictions, axis=-1)

    with open(pred_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for pred in predictions:
            writer.writerow([pred])


# [[tn, fp],
#  [fn, tp]]
metric = evaluate.load("confusion_matrix.py")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


class CustomTrainer(Trainer):
    def __init__(self, training_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_args = training_args

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.pop("labels")
        loss_ce = nn.functional.cross_entropy(logits, labels)

        labels = self.training_args.label_smoothing * labels.float()
        probs = logits.log_softmax(dim=1).exp()[:, 1]
        cardinality = torch.sum(probs + labels)
        difference = torch.sum(torch.abs(probs - labels))
        intersection = 0.5 * (cardinality - difference)
        fp = torch.sum(probs) - intersection
        fn = torch.sum(labels) - intersection
        tversky = intersection / (
            intersection
            + self.training_args.fp_ratio * fp
            + self.training_args.fn_ratio * fn
        )
        loss_tversky = 1 - tversky

        loss = (
            1 - self.training_args.tversky_ratio
        ) * loss_ce + self.training_args.tversky_ratio * loss_tversky

        return (loss, outputs) if return_outputs else loss


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        num_labels=model_args.num_labels,
        local_files_only=True,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # load data and construct Dataset
    print("Loading data...")
    data_dict = load_dataset(data_args.data_path)
    vicuna_data_dict = load_dataset(data_args.vicuna_data_path)
    wizardlm_data_dict = load_dataset(data_args.wizardlm_data_path)

    data_module = make_supervised_data_module(
        data_dict, train_val_ratio=data_args.train_val_ratio
    )

    vicuna_dataset = SupervisedDataset(vicuna_data_dict)
    wizardlm_dataset = SupervisedDataset(wizardlm_data_dict)

    # construct trainer
    trainer = CustomTrainer(
        training_args=training_args,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    vicuna_results = trainer.predict(vicuna_dataset)
    print(vicuna_results.metrics)
    vicuna_pred_path = os.path.join(
        training_args.output_dir, data_args.vicuna_pred_path
    )
    save_predictions(vicuna_results, vicuna_pred_path)

    wizardlm_results = trainer.predict(wizardlm_dataset)
    print(wizardlm_results.metrics)
    wizardlm_pred_path = os.path.join(
        training_args.output_dir, data_args.wizardlm_pred_path
    )
    save_predictions(wizardlm_results, wizardlm_pred_path)

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
