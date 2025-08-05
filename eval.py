import re
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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
    return answer.strip()

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


model_name = '/data2/alex/verifiers/outputs_s3/outputs_capacityblock0/Qwen2.5-7B-Instruct-gsm8k-gamma0.99999975-seed44-constantlr-1epoch-capacityblock0/checkpoint-935'

llm = LLM(
    model=model_name,
    tensor_parallel_size=1,
    dtype="auto",
    trust_remote_code=True,
    max_model_len=1024,
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

sampling_params = SamplingParams(temperature=0.0, top_p = 1.0, top_k = 0, max_tokens=786, seed = 42)
outputs = llm.generate(prompts, sampling_params)


correct = 0
for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text
    predicted_answer = extract_xml_answer(generated_text)
    ground_truth = eval_data[i]["other_answer"]
    #print(ground_truth, predicted_answer)
    #if int(predicted_answer) == int(ground_truth):
    if predicted_answer == ground_truth:
        correct += 1
    print(correct / (i+1))

response_lengths = []
for output in outputs:
    # Get the first completion for the prompt
    first_completion = output.outputs[0]

    # Get the number of tokens in this completion
    num_tokens = len(first_completion.token_ids)

    # Add it to our list
    response_lengths.append(num_tokens)

print(sum(response_lengths) / len(response_lengths))

