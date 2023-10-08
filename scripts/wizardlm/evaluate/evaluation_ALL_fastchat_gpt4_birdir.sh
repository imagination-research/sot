for model in llama_v2_7b llama_v2_13b openchat stable-vicuna ultralm_13b vicuna7b vicuna7bv13 vicuna13bv13 vicuna33bv13 gpt3.5 claude_slack gpt4
do

python sot/fastchat_evaluation_bidir.py \
--model openai \
--output-folder results/wizardlm/wizardlm_${model}_outline/compare_to_naive_by_gpt4 \
--answer-1-file results/wizardlm/wizardlm_${model}_outline/data.csv \
--answer-2-file results/wizardlm/wizardlm_${model}_naive/data.csv \
--template "[Question]
{question}

[The Start of Assistant 1's Answer]
{answer_1}

[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer_2}

[The End of Assistant 2's Answer]

[System]
{prompt}

" \
--prompt "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment." \
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
