import os
import subprocess


MODELS = [
    ("llama_v2_7b", "meta-llama/Llama-2-7b-chat-hf", 1),
    ("llama_v2_13b", "meta-llama/Llama-2-13b-chat-hf", 2),
    ("openchat", "openchat/openchat", 3),
    ("stable-vicuna", "TheBloke/stable-vicuna-13B-HF", 3),
    ("ultralm_13b", "TheBloke/UltraLM-13B-fp16", 3),
    (
        "vicuna7b",
        "lmsys/vicuna-7b-v1.1",
        1,
    ),
    ("vicuna7bv13", "lmsys/vicuna-7b-v1.3", 1),
    ("vicuna13bv13", "lmsys/vicuna-13b-v1.3", 2),
    ("vicuna33bv13", "lmsys/vicuna-33b-v1.3", 5),
]

OUTPUT_FOLDER = "results/vicuna/vicuna_{model_name}_naive"

COMMAND = (
    "python sot/main.py --model fastchat --scheduler naive "
    "--data-path data/vicuna/data.csv --output-folder {output_folder} "
    "--model-path {model_path} --num-gpus {num_gpus}"
)


if __name__ == "__main__":
    for model_name, path, num_gpus in MODELS:
        output_folder = OUTPUT_FOLDER.format(model_name=model_name)
        if os.path.exists(output_folder):
            print(f"Skipping {model_name}")
        else:
            print(f"Running {model_name}")
            command = COMMAND.format(
                output_folder=output_folder, model_path=path, num_gpus=num_gpus
            )
            print(command)
            process = subprocess.Popen(command.split())
            process.wait()
