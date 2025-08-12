name='Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed42-beta0.1-8steps-1epoch-temp0.6-182'

python eval.py \
  --model_name "/data2/alex/verifiers/outputs/${name}/checkpoint-935" \
  --runs 30 \
  --max_model_len 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_tokens 786 \
  > "eval_results/${name}.txt"


name='Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed42-beta0.1-16steps-1epoch-193'

python eval.py \
  --model_name "/data2/alex/verifiers/outputs/${name}/checkpoint-935" \
  --runs 30 \
  --max_model_len 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_tokens 786 \
  > "eval_results/${name}.txt" 


name='Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed42-beta0.1-16steps-1epoch-temp0.6-182'

python eval.py \
  --model_name "/data2/alex/verifiers/outputs/${name}/checkpoint-935" \
  --runs 30 \
  --max_model_len 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_tokens 786 \
  > "eval_results/${name}.txt" 


name='Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed42-beta0.1-32sync-1epoch-182'

python eval.py \
  --model_name "/data2/alex/verifiers/outputs/${name}/checkpoint-935" \
  --runs 30 \
  --max_model_len 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_tokens 786 \
  > "eval_results/${name}.txt"


name='Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed42-beta0.1-64sync-1epoch-182'

python eval.py \
  --model_name "/data2/alex/verifiers/outputs/${name}/checkpoint-935" \
  --runs 30 \
  --max_model_len 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_tokens 786 \
  > "eval_results/${name}.txt" 


name='Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed42-beta0.4-4steps-1epoch-193'

python eval.py \
  --model_name "/data2/alex/verifiers/outputs/${name}/checkpoint-935" \
  --runs 30 \
  --max_model_len 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_tokens 786 \
  > "eval_results/${name}.txt" 


  name='Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed42-beta0.4-8steps-1epoch-193'

python eval.py \
  --model_name "/data2/alex/verifiers/outputs/${name}/checkpoint-935" \
  --runs 30 \
  --max_model_len 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_tokens 786 \
  > "eval_results/${name}.txt"


name='Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed42-beta0.4-8steps-1epoch-temp0.6-182'

python eval.py \
  --model_name "/data2/alex/verifiers/outputs/${name}/checkpoint-935" \
  --runs 30 \
  --max_model_len 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_tokens 786 \
  > "eval_results/${name}.txt" 

  name='Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed42-beta0.4-16steps-1epoch-193'

python eval.py \
  --model_name "/data2/alex/verifiers/outputs/${name}/checkpoint-935" \
  --runs 30 \
  --max_model_len 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_tokens 786 \
  > "eval_results/${name}.txt" 


name='Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed42-beta0.4-32sync-1epoch-182'

python eval.py \
  --model_name "/data2/alex/verifiers/outputs/${name}/checkpoint-935" \
  --runs 30 \
  --max_model_len 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_tokens 786 \
  > "eval_results/${name}.txt" 

  name='Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed42-beta0.4-64sync-1epoch-182'

python eval.py \
  --model_name "/data2/alex/verifiers/outputs/${name}/checkpoint-935" \
  --runs 30 \
  --max_model_len 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_tokens 786 \
  > "eval_results/${name}.txt" 


name='Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed42-beta0.6-16steps-1epoch-182'

python eval.py \
  --model_name "/data2/alex/verifiers/outputs/${name}/checkpoint-935" \
  --runs 30 \
  --max_model_len 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_tokens 786 \
  > "eval_results/${name}.txt" 

  name=''

python eval.py \
  --model_name "/data2/alex/verifiers/outputs/${name}/checkpoint-935" \
  --runs 30 \
  --max_model_len 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_tokens 786 \
  > "eval_results/${name}.txt" 


name='Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed42-beta0.6-32sync-1epoch-182'

python eval.py \
  --model_name "/data2/alex/verifiers/outputs/${name}/checkpoint-935" \
  --runs 30 \
  --max_model_len 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_tokens 786 \
  > "eval_results/${name}.txt"