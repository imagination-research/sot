for model in llama_v2_7b llama_v2_13b openchat stable-vicuna ultralm_13b vicuna7b vicuna7bv13 vicuna13bv13 vicuna33bv13 gpt3.5 claude_slack gpt4
do

python sot/fastchat_evaluation_for_vicuna_bidir.py \
--model openai \
--output-folder results/vicuna/vicuna_${model}_outline/compare_to_naive_by_gpt4 \
--answer-1-file results/vicuna/vicuna_${model}_outline/data.csv \
--answer-2-file results/vicuna/vicuna_${model}_naive/data.csv \
--api-type open_ai \
--api-base https://api.openai.com/v1 \
--temperature 0.2 \
--max-tokens 5000 \
--top-p 0.95 \
--frequency-penalty 0 \
--presence-penalty 0 \
--api-model gpt-4-0613 \
--system-message 'You are a helpful and precise assistant for checking the quality of the answer.' \
--timeout 240

done