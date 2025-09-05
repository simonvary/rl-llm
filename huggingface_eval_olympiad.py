import re
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from math_verify import parse, verify
import multiprocessing as mp
import os

SYSTEM_PROMPT = """You must reply in EXACTLY this XML:

<reasoning>
...
</reasoning>
<answer>
...
</answer>

Rules:
- All text must be wrapped inside a <reasoning> </reasoning> or <answer> </answer> tag. 
"""

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on AIME25 with vLLM.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--max_model_len", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", default=-1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def extract_xml_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"<answer>\s*(.*)$", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

def extract_numerical_answer(answer_text):
    match = re.search(r"#### ([-\d,]+)", answer_text)
    if match:
        return int(match.group(1).replace(",", ""))
    return None

def main():
    args = get_args()

    # vLLM sometimes emits tokenizer parallelism warnings; silence if you like
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Base tokenizer (only for vLLM's internal tokenizer if you need to override)
    BASE_TOKENIZER_ID = "microsoft/phi-4"

    # Normalize top_k: vLLM expects int or None, not -1
    top_k = args.top_k

    # Build the engine INSIDE main / after spawn method is set
    llm = LLM(
        model=args.model_name,
        tokenizer=BASE_TOKENIZER_ID,
        tensor_parallel_size=1,
        dtype="auto",               # OK: avoids the deprecated torch_dtype kw
        # trust_remote_code only matters for custom model code; okay to keep here
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.95,
        enforce_eager=False,        # use FA2 if available
    )

    # Use the model repo’s tokenizer to format chat prompts correctly
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Prepare data
    data = load_dataset("math-ai/olympiadbench")["test"]
    eval_data = []
    for item in data:
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        eval_data.append({
            "question": item["question"],
            "prompt": formatted_prompt,
            "answer": item["final_answer"][0],
            "numerical_answer": item["final_answer"][0],
            "other_answer": item["final_answer"][0],
        })

    prompts = [it["prompt"] for it in eval_data]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop=["</answer>"],
    )

    runs = args.runs
    mean_correct = 0.0
    mean_length = 0.0

    for run in tqdm(range(runs)):
        outputs = llm.generate(prompts, sampling_params)

        correct = 0
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            answer = parse(extract_xml_answer(generated_text))
            gold = parse(eval_data[i]["other_answer"])
            if verify(gold, answer):
                correct += 1
        mean_correct += correct / (i + 1)

        response_lengths = []
        for output in outputs:
            first_completion = output.outputs[0]
            num_tokens = len(first_completion.token_ids)
            response_lengths.append(num_tokens)
        mean_length += sum(response_lengths) / len(response_lengths)

    print("Model:", args.model_name)
    print("Mean Correct:", mean_correct / runs)
    print("Mean Length:", mean_length / runs)

if __name__ == "__main__":
    # Ensure the same start method everywhere; vLLM may internally rely on spawn.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
