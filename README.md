# Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation

**[[website](https://sites.google.com/view/sot-llm/home)]**
**[[paper (ICLR 2024)](https://openreview.net/forum?id=mqVgBbNCm9)]**
**[[paper (arXiv)](https://arxiv.org/abs/2307.15337)]**
**[[code](https://github.com/imagination-research/sot)]**
**[[blog](https://www.microsoft.com/en-us/research/blog/skeleton-of-thought-parallel-decoding-speeds-up-and-improves-llm-output/)]**

This work aims at decreasing the end-to-end generation latency of large language models (LLMs). One of the major causes of the high generation latency is the sequential decoding approach adopted by almost all state-of-the-art LLMs. In this work, motivated by the thinking and writing process of humans, we propose Skeleton-of-Thought (SoT), which first guides LLMs to generate the skeleton of the answer, and then conducts parallel API calls or batched decoding to complete the contents of each skeleton point in parallel. Not only does SoT provide considerable speed-ups across 12 LLMs, but it can also potentially improve the answer quality on several question categories. To make the overall solution more practical, an extension, SoT with Router (SoT-R), employs a GPT-4-prompting router or a trained RoBERTa router to only trigger SoT for suitable questions. SoT is an initial attempt at data-centric optimization for inference efficiency, and further underscores the potential of pushing LLMs to think more like a human for answer quality.


If you find this repository or paper useful, you can cite
```
@inproceedings{
ning2024skeletonofthought,
title={Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation},
author={Xuefei Ning and Zinan Lin and Zixuan Zhou and Zifu Wang and Huazhong Yang and Yu Wang},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=mqVgBbNCm9}
}
```

The repo is organized as follows.
* The SoT implementation is under [`sot/`](sot/).
* The SoT prompts are given under [`prompts/`](prompts/). For example, `sot_opensource.json` is used for all open-source models, and `sot_gpt4` is used for the GPT-4 API.
* The processed data is under [`data/`](data/).
* The scripts under [`scripts/`](scripts/) are used to dump and evaluate the results.
* The Gradio demo code is under [`demo/`](demo/). The demo is built based on the FastChat demo code.

## Contents
- [Install](#install)
- [Test SoT with Gradio Demo](#test-sot-with-gradio-demo)
- [Test SoT in the Console](#test-sot-in-the-console)
- [Evaluate SoT](#evaluate-sot)
- [Develop SoT](#develop-sot)
- [Acknowledgement](#acknowledgement)


## Install
```pip install -e .```

We recommend using Python 3.8 to 3.10.

Some required environment variables/setups for using the API-based models:
* For GPT4, the script by default uses **OpenAI API**. The API key should be provided by `export OPENAI_API_KEY=<API key>`. 
* For GPT-3.5, the script by default uses **Azure OpenAI API**. The API key, engine, and API base should be provided by `export OPENAI_API_KEY=<API key>`, `export ENGINE=<engine>`, and `export API_BASE=<API base>`.
  > Note that GPT-4 can also use **Azure OpenAI API**, and GPT-3.5 can also use **OpenAI API**, by modifying the command line arguments accordingly. 
* For Claude, please refer to [Claude setup guide](claude_setup_guide.md).

## Test SoT with Gradio Demo
The SoT gradio demo for open-source models can be started by running the following commands under the [`demo/`](demo/) directory:

1. Launch the controller
  ```
  python controller.py
  ```
2. Launch the model workers
  - Lauch a model worker that conducts normal decoding on GPU 0.
    ```
    CUDA_VISIBLE_DEVICES=0 python model_worker.py --model-path ${MODEL_NAME} --controller http://0.0.0.0:21001 --port 31000 --worker http://0.0.0.0:31000
    ```
  - Launch a model worker that conducts SoT-R decoding (with RoBERTa router) on GPU 1.
    ```
    CUDA_VISIBLE_DEVICES=1 python model_worker.py --model-path ${MODEL_NAME} --controller http://0.0.0.0:21001 --port 31001 --worker http://0.0.0.0:31001 --sot ../prompts/sot_opensource.json --sotr ${ROUTER_MODEL}
    ```
    The trained router model can be downloaded from [this Google Drive](https://drive.google.com/file/d/1LxEsH9NFwj41wBz8tnT_hwn5LbW7aaL5/view?usp=sharing).
  - Note that we recommend directly using SoT-R instead of the plain SoT. But if one wants to trigger SoT for all questions, please use the following command instead:
    ```
    CUDA_VISIBLE_DEVICES=1 python model_worker.py --model-path ${MODEL_NAME} --controller http://0.0.0.0:21001 --port 31002 --worker http://0.0.0.0:31002 --sot ../prompts/sot_opensource.json
    ```
3. Launch the Gradio web demo
  ```
  python gradio_web_server_multi.py
  ```

## Test SoT in the Console
Besides chatting with SoT using the web demo, another convenient way to check how SoT works on specific questions is to use the `sot/prompt_eng_main.py` helper program. In the interactive session popped by this helper program, one can issue questions saved in data files to SoT and check the outputs in the console conveniently. See [this section](#manually-tune-the-sot-prompts) for more details.

## Evaluate SoT
### Prepare the dataset
Vicuna-80, WizardLM, and LIMA data are provided under [`data/`](data/) and are ready to use. The pre-processing scripts for getting the data are also attached (`create_dataset.py` in each folder) for reference.

### Dump the answers of SoT and Normal decoding
We put the answer dumping scripts for the Vicuna-80 and WizardLM datasets under [`scripts/vicuna/dump/`](scripts/vicuna/dump/) and [`scripts/wizardlm/dump/`](scripts/wizardlm/dump/).

For example, to dump SoT answers of all open-source models, we can run
```
python scripts/vicuna/dump/opensource_outline.py
```

To dump the normal sequential decoding answers of GPT-3.5, we can run
```
./scripts/vicuna/dump/gpt3.5_naive.sh
```

### Evaluate the answer quality
We put the evaluation scripts for the Vicuna-80 and WizardLM datasets under [`scripts/vicuna/evaluate/`](scripts/vicuna/evaluate/) and [`scripts/wizardlm/evaluate/`](scripts/wizardlm/evaluate/).

The evaluation scripts use the comparison prompts provided by Fastchat or LLMZoo to prompt a GPT-4 judge to compare the quality of two answers.  Please provide the OpenAI API key by `export OPENAI_API_KEY=<API key>` before running the scripts.

> Note:
> We did not use the system prompt except for the LLaMA-2 models when conducting open-source model evaluation in our paper (for both normal decoding and SoT decoding). This is because we use an [older FastChat version](https://github.com/lm-sys/FastChat/tree/f1f2294a66956b340c577fab2751d86f45e71099) for the evaluation in the paper. As our code removes the template messages in the conversation template before querying the model, the system prompt will be removed as template messages in the old FastChat version. Nevertheless, in this code repository, we use a newer version of FastChat (v0.2.26). Consequently, running SoT with the current code will use the system prompt for all open-source models.

The above evaluation is only for SoT (without routers). Please refer to [`prompts/router_gpt4.json`](prompts/router_gpt4.json) for the prompt we use for SoT-R with Prompting Router (using GPT-4), and [this section](#train-the-router-for-sot-r) for details about SoT-R with Trained Router (using RoBERTa).


## Develop SoT
### Manually tune the SoT prompts
`sot/prompt_eng_main.py` is a helper program to ease manual prompt tuning. Use `bash scripts/debug_prompt.sh <model name or path>` to run the script. This will pop an interactive session in which you can run the following commands:

1. `usedata <data filepath>` to load data from the given file path (default: `data/vicuna/data.csv`)
2. `useprompt <prompt filepath>` to change the SoT prompt templates (default: `prompts/sot_opensource.json`)
3. `usenaiveprompt <prompt filepath>` to change the normal prompt template (default to use only the question)
4.  (1) `test <ind>` to test SoT decoding for the `<ind>`-th question; (2) `test naive <ind>` to test normal decoding; (3) `test batch_outline <ind>` to test SoT decoding with batched point expansion.
    * The model outputs will be streamed onto the console (by enabling `--stream` argument to `sot/prompt_eng_main.py`). Note that when using `test <ind>`, the expansion of multiple points is conducted sequentially. When using `test batch_outline <ind>`, the expansion of multiple points is conducted with batch inference, but we do not support streaming the parallel expansion outputs to the console (to check the streaming effect, use the Gradio Web Demo), so one has to wait until the point-expanding completion to see the results.
    * After the completion, statistics will also be printed.
    * At any time during the generation, one can push Ctrl+C to abort the generation to go back to the interactive session.
5. `exit` to exit the session

> Note:
> 1. We mainly use this program to help engineer the prompt for the open-source models, and didn't test it with the API-based models.
> 2. Any other command-line arguments for the model can be fed as the arguments to this script. For example, as testing a 13B model on RTX 3090 with FP16 inference requires two GPUs, we can run
> ```bash scripts/debug_prompt.sh meta-llama/Llama-2-13b-chat-hf --num-gpus 2```

### Train the router for SoT-R
Preprocess router data and train the RoBERTa router as follows (scripts in [`sot/train/`](sot/train/)):

1. Preprocess the router data for Vicuna-80, WizardLM, and LIMA:
  ```
  python offline_prepare_router_data.py \
    --data_path "../../data/lima/router.csv" \
    --output_data_path "lima_router.pkl"
  ```
2. Train the router on LIMA and test on Vicuna-80 and WizardLM:
  ```
  python train_router.py
  ```

The predicted results will be saved as `vicuna_router_pred.csv` and `wizardlm_router_pred.csv`.

Our trained router model can be found on [this Google Drive](https://drive.google.com/file/d/1LxEsH9NFwj41wBz8tnT_hwn5LbW7aaL5/view?usp=sharing).

Our manual labels of whether each question should use SoT are provided in `data/*/router.csv`.

## Acknowledgement
During the development of SoT, we use and refer to the amazing work of [FastChat](https://github.com/lm-sys/FastChat) and [Hugging Face transformer package](https://github.com/huggingface/transformers/).
