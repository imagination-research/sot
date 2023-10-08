for model in llama_v2_7b llama_v2_13b openchat stable-vicuna ultralm_13b vicuna7b vicuna7bv13 vicuna13bv13 vicuna33bv13 gpt3.5 claude_slack gpt4
do

RESULT_FOLDER=results/vicuna/vicuna_${model}_outline/compare_to_naive_by_gpt4
ANSWER_FILE_1=results/vicuna/vicuna_${model}_outline/data.csv
ANSWER_FILE_2=results/vicuna/vicuna_${model}_naive/data.csv

python sot/llm_zoo_evaluation_alldir.py \
--model openai \
--output-data-filename llm_zoo_evaluation_general.csv \
--log-filename llm_zoo_evaluation_general_log.log \
--output-folder ${RESULT_FOLDER} \
--answer-file ${ANSWER_FILE_1}  \
--answer-file ${ANSWER_FILE_2}  \
--api-type open_ai \
--api-base https://api.openai.com/v1 \
--temperature 0.0 \
--max-tokens 5000 \
--top-p 0.95 \
--frequency-penalty 0 \
--presence-penalty 0 \
--api-model gpt-4-0613 \
--system-message 'You are a helpful and precise assistant for checking the quality of the answer.' \
--prompt "We would like to request your feedback on the performance of {num_str} AI assistants in response to the user question displayed above.
Please evaluate the given four aspects: helpfulness, relevance, accuracy, level of details of their responses.
Please first clarify how each response achieves each aspect respectively.
Then, provide a comparison on the overall performance among Assistant 1 - Assistant {num}, and you need to clarify which one is better than or equal to another. Avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
In the last line, order the {num_str} assistants. Please output a single line ordering Assistant 1 - Assistant {num}, where '>' means 'is better than' and '=' means 'is equal to'. The order should be consistent to your comparison. If there is not comparision that one is better, it is assumed they have equivalent overall performance ('=')."
--timeout 240

python sot/llm_zoo_evaluation_alldir.py \
--model openai \
--output-data-filename llm_zoo_evaluation_relevance.csv \
--log-filename llm_zoo_evaluation_relevance_log.log \
--output-folder ${RESULT_FOLDER} \
--answer-file ${ANSWER_FILE_1}  \
--answer-file ${ANSWER_FILE_2}  \
--api-type open_ai \
--api-base https://api.openai.com/v1 \
--temperature 0.0 \
--max-tokens 5000 \
--top-p 0.95 \
--frequency-penalty 0 \
--presence-penalty 0 \
--api-model gpt-4-0613 \
--system-message 'You are a helpful and precise assistant for checking the quality of the answer.' \
--prompt "Relevance: The response should be closely related to the question and answer the question accurately with sufficient details without repetition or redundancy. The more relevant they are, the better.
Please evaluate the relevance of {num_str} AI assistants in response to the user question displayed above.
Please first clarify how each response addresses the question and whether it is accurate respectively.
Then, provide a comparison on relevance among Assistant 1 - Assistant {num}, and you need to clarify which one is more relevant than or equal to another. Avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
In the last line, order the {num_str} assistants. Please output a single line ordering Assistant 1 - Assistant {num}, where '>' means 'is better than' and '=' means 'is equal to'. The order should be consistent to your comparison. If there is not comparision that one is more relevant, it is assumed they have equivalent relevance ('=')." \
--timeout 240

python sot/llm_zoo_evaluation_alldir.py \
--model openai \
--output-data-filename llm_zoo_evaluation_diversity.csv \
--log-filename llm_zoo_evaluation_diversity_log.log \
--output-folder ${RESULT_FOLDER} \
--answer-file ${ANSWER_FILE_1}  \
--answer-file ${ANSWER_FILE_2}  \
--api-type open_ai \
--api-base https://api.openai.com/v1 \
--temperature 0.0 \
--max-tokens 5000 \
--top-p 0.95 \
--frequency-penalty 0 \
--presence-penalty 0 \
--api-model gpt-4-0613 \
--system-message 'You are a helpful and precise assistant for checking the quality of the answer.' \
--prompt "Diversity: The response should be comprehensive and provide a range of information that is not limited to a single perspective. More perspectives are better.
Please evaluate the diversity of {num_str} AI assistants in response to the user question displayed above.
Please first clarify which perspectives and aspects they consider and the diversity they explore respectively.
Then, provide a comparison on diversity among Assistant 1 - Assistant {num}, and you need to clarify which one is more diverse than or equal to another. Avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
In the last line, order the {num_str} assistants. Please output a single line ordering Assistant 1 - Assistant {num}, where '>' means 'is better than' and '=' means 'is equal to'. The order should be consistent to your comparison. If there is not comparision that one is more diverse, it is assumed they have equivalent diversity ('=')." \
--timeout 240

python sot/llm_zoo_evaluation_alldir.py \
--model openai \
--output-data-filename llm_zoo_evaluation_coherence.csv \
--log-filename llm_zoo_evaluation_coherence_log.log \
--output-folder ${RESULT_FOLDER} \
--answer-file ${ANSWER_FILE_1}  \
--answer-file ${ANSWER_FILE_2}  \
--api-type open_ai \
--api-base https://api.openai.com/v1 \
--temperature 0.0 \
--max-tokens 5000 \
--top-p 0.95 \
--frequency-penalty 0 \
--presence-penalty 0 \
--api-model gpt-4-0613 \
--system-message 'You are a helpful and precise assistant for checking the quality of the answer.' \
--prompt "Coherence: The response should be coherent and flow logically from one point to the next that is easy to read and understand without major gaps or inconsistencies. The more coherent they are, the better.
Please evaluate the coherence of {num_str} AI assistants in response to the user question displayed above.
Please first clarify to what degree each response flows smoothly from one point to another respectively.
Then, provide a comparison on coherence among Assistant 1 - Assistant {num}, and you need to clarify which one is more coherent than or equal to another. Avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
In the last line, order the {num_str} assistants. Please output a single line ordering Assistant 1 - Assistant {num}, where '>' means 'is better than' and '=' means 'is equal to'. If there is not comparision that one is more coherent, it is assumed they have equivalent coherence ('=')." \
--timeout 240

python sot/llm_zoo_evaluation_alldir.py \
--model openai \
--output-data-filename llm_zoo_evaluation_immersion.csv \
--log-filename llm_zoo_evaluation_immersion_log.log \
--output-folder ${RESULT_FOLDER} \
--answer-file ${ANSWER_FILE_1}  \
--answer-file ${ANSWER_FILE_2}  \
--api-type open_ai \
--api-base https://api.openai.com/v1 \
--temperature 0.0 \
--max-tokens 5000 \
--top-p 0.95 \
--frequency-penalty 0 \
--presence-penalty 0 \
--api-model gpt-4-0613 \
--system-message 'You are a helpful and precise assistant for checking the quality of the answer.' \
--prompt "Immersion: The response should act like the assigned role using the tone, manner and vocabulary the role would use. The more assistant-like tones, the worse. The more in-character, the better.
Please evaluate the Immersion of {num_str} AI assistants in response to the user question displayed above.
Please first clarify how they pretend to be the role and to what extend they have successfully simulated it respectively.
Then, provide a comparison on Immersion among Assistant 1 - Assistant {num}, and you need to clarify which one is more in-character than or equal to another. Avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
In the last line, order the {num_str} assistants. Please output a single line ordering Assistant 1 - Assistant {num}, where '>' means 'is better than' and '=' means 'is equal to'. The order should be consistent to your comparison. If there is not comparision that one is more in-character, it is assumed they have equivalent immersion ('=')." \
--timeout 240

python sot/llm_zoo_evaluation_alldir.py \
--model openai \
--output-data-filename llm_zoo_evaluation_integrity.csv \
--log-filename llm_zoo_evaluation_integrity_log.log \
--output-folder ${RESULT_FOLDER} \
--answer-file ${ANSWER_FILE_1}  \
--answer-file ${ANSWER_FILE_2}  \
--api-type open_ai \
--api-base https://api.openai.com/v1 \
--temperature 0.0 \
--max-tokens 5000 \
--top-p 0.95 \
--frequency-penalty 0 \
--presence-penalty 0 \
--api-model gpt-4-0613 \
--system-message 'You are a helpful and precise assistant for checking the quality of the answer.' \
--prompt "Integrity: The response should be ethical and moral soundness and adherence to principles and values, while avoiding stereotypes, offensive language, misleading information, or harmful suggestions that can negatively impact individuals, groups, or society. The more immoral or harmful language they produce, the worse.
Please integrity the relevance of {num_str} AI assistants in response to the user question displayed above.
Please first clarify whether each response produces inmmoral or harmful language, and to what degree they are unethical respectively.
Then, provide a comparison on integrity among Assistant 1 - Assistant {num}, and you need to clarify which one is less moral than or equal to another. Avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
In the last line, order the {num_str} assistants. Please output a single line ordering Assistant 1 - Assistant {num}, where '>' means 'is better than' and '=' means 'is equal to'. The order should be consistent to your comparison. If there is not comparision that one is less moral, it is assumed they have equivalent integrity ('=')." \
--timeout 240

done
