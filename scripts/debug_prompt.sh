model_path=${1}
shift 1
model_suffix=${model_path##*/}
cur_time=$(date +"%Y-%m-%d--%H-%M")
res_file=results/${model_suffix}-${cur_time}.log

echo "Test ${model_path}; Log will be saved to ${res_file}"

python sot/prompt_eng_main.py --model fastchat --output-log ${res_file} --model-path ${model_path} --stream $@
