import re
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from math_verify import parse, verify




# ----------------------------------------------------------------------------
# Argument Parser
# ----------------------------------------------------------------------------
def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a model using GRPOTrainer with configurable arguments.")

    # Model and Tokenizer
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="The name of the model to evaluate.")
    parser.add_argument("--runs", type=int, default=30, help="Number of runs to average the results over.")
    parser.add_argument("--max_model_len", type=int, default=1024, help="Maximum model length.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter.")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling parameter.")
    parser.add_argument("--max_tokens", type=int, default=786, help="Maximum number of tokens to generate in each response.")


    return parser.parse_args()
args = get_args()


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_inter_answer(text: str) -> int:
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])
    return 0

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    #print(answer)
    return answer

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

def extract_numerical_answer(answer_text):
    # GSM8K answers end with #### followed by the numerical answer
    match = re.search(r"#### ([-\d,]+)", answer_text)
    if match:
        # Remove commas and convert to int
        return int(match.group(1).replace(",", ""))
    return None


#model_name = 'data2/alex/verifiers/outputs/Qwen2.5-0.5B-Instruct-gsm8k-gamma1.0-seed42-beta0.1-8steps-1epoch-193/checkpoint-935'  # Example model name, replace with your actual model path
model_name = args.model_name

llm = LLM(
    model=model_name,
    tensor_parallel_size=1,
    dtype="auto",
    trust_remote_code=True,
    max_model_len=args.max_model_len,
    gpu_memory_utilization=0.95,
    enforce_eager=False,  # Use Flash Attention 2
)

data = load_dataset("openai/gsm8k", "main")["test"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
eval_data = []
for i, item in enumerate(data):
    # Create the chat structure, same as in training
    chat = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': item["question"]},
    ]
    
    # Apply the template to get the correctly formatted prompt string
    formatted_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True # Adds the prompt for the assistant's turn
    )
    
    proccessed = {
        "question": item["question"],
        "prompt": formatted_prompt, # Use the correctly formatted prompt
        "answer": item["answer"],
        "numerical_answer": extract_numerical_answer(item["answer"]),
        "other_answer": extract_hash_answer(item["answer"]),
    }
    eval_data.append(proccessed)

prompts = [item["prompt"] for item in eval_data]




sampling_params = SamplingParams(temperature=args.temperature, top_p = args.top_p, top_k = args.top_k, max_tokens=args.max_tokens, seed = 42)
runs = args.runs
mean_correct = 0
mean_length = 0
for run in tqdm(range(runs)):

    outputs = llm.generate(prompts, sampling_params)


    correct = 0
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        answer = parse(extract_xml_answer(generated_text))
        gold = parse(eval_data[i]["other_answer"])
        if verify(gold,answer) == True:
            correct += 1
    #print(correct / (i+1))
    mean_correct += correct / (i+1)

    response_lengths = []
    for output in outputs:
        # Get the first completion for the prompt
        first_completion = output.outputs[0]

        # Get the number of tokens in this completion
        num_tokens = len(first_completion.token_ids)

        # Add it to our list
        response_lengths.append(num_tokens)

    #print(sum(response_lengths) / len(response_lengths))
    mean_length += sum(response_lengths) / len(response_lengths)

print("Model:", model_name)
print("Mean Correct:", mean_correct / runs)
print("Mean Length:", mean_length / runs)
