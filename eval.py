# #!/usr/bin/env python3
# """
# Evaluate a GRPO-fine-tuned model on GSM8K and report

#     • running pass@1 / mean-completion-length every N examples
#     • final pass@1 / mean-completion-length

# Generation hyper-parameters and stop-logic match those of GRPOTrainer.
# """
# import re, argparse
# from datasets import load_dataset, Dataset
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     StoppingCriteria,
#     StoppingCriteriaList,
# )
# import torch
# from tqdm import tqdm
# from pathlib import Path
# from accelerate.utils import fsdp_utils

# # ───────────────────────────────── prompt template ───────────────────────────
# SYSTEM_PROMPT = """
# Respond in the following format:
# <reasoning>
# ...
# </reasoning>
# <answer>
# ...
# </answer>
# """

# # SYSTEM_PROMPT = """
# # You are an intelligent math assistant designed to solve math problems that require careful reasoning.

# # When solving a math problem, you should:
# # 1. Break the problem down into steps
# # 2. Reason carefully through each step
# # 3. Provide a clear final answer in simplified form

# # Format your entire response using one these XML tags:
# # <reasoning>
# # Think step-by-step about how to solve the math problem, explaining the approach clearly.
# # </reasoning>
# # <answer>
# # Your final answer to the math problem, in simplified form.
# # </answer>

# # First use the <reasoning> tag to think through the problem. When you're ready to provide the final answer, use the <answer> tag.
# # """

# # ────────────────────────────────── helpers ──────────────────────────────────
# _PUNCT_PAT = re.compile(r"\s+")

# def extract_xml_answer(text: str) -> str:
#     answer = text.split("<answer>")[-1]
#     answer = answer.split("</answer>")[0]
#     return answer.strip()

# def extract_hash_answer(text: str) -> str | None:
#     if "####" not in text:
#         return None
#     return text.split("####")[1].strip().replace(",", "").replace("$", "")


# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     q = prompts[0][-1]['content']
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     #print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
#     return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# def get_gsm8k(split: str = "test") -> Dataset:
#     ds = load_dataset("openai/gsm8k", "main")[split]
#     def _wrap(ex):
#         return {
#             "prompt": [
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user",   "content": ex["question"]},
#             ],
#             "answer": extract_hash_answer(ex["answer"]),
#         }
#     return ds.map(_wrap)

# # ───────────────────────────── custom stopping ───────────────────────────────
# class StopOnStrings(StoppingCriteria):
#     """Stop generation once any of the `stop_ids` appears."""
#     def __init__(self, stop_ids: list[list[int]]):
#         super().__init__()
#         self.stop_ids = stop_ids

#     def __call__(self, input_ids, scores, **kwargs):
#         seq = input_ids[0].tolist()
#         for pat in self.stop_ids:
#             if len(seq) >= len(pat) and seq[-len(pat):] == pat:
#                 return True
#         return False

# # ─────────────────────────────────── main ────────────────────────────────────
# def main():
    
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model-dir",      type=str,
#                     default="/data2/alex/verifiers/outputs/Qwen-1.5B-discount0/checkpoint-7600")
#     ap.add_argument("--split",          choices=["train", "test"], default="test")
#     ap.add_argument("--device",         default="cuda")
#     ap.add_argument("--max-token-len",  type=int, default=786,
#                     help="maximum number of new tokens to generate")
#     ap.add_argument("--k",              type=int, default=1,
#                     help="number of samples (generations) per query")
#     ap.add_argument("--limit",          type=int, default=9999)
#     ap.add_argument("--temperature",    type=float, default=0.0)
#     ap.add_argument("--log-interval",   type=int, default=1,
#                     help="print running metrics every N examples")
#     ap.add_argument("--checkpoint",     type=bool, default=False)
#     args = ap.parse_args()
    

#     # Load model / tokenizer
#     if args.checkpoint == True:
#         ckpt_dir   = Path("/data2/.../checkpoint-100")       # folder with pytorch_model_fsdp_0/
#         base_model = "Qwen/Qwen2.5-1.5B-Instruct"            # original model repo

#         tokenizer = AutoTokenizer.from_pretrained(base_model)
#         model     = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)

#         # ① Load the consolidated state-dict
#         state_dict = fsdp_utils.load_fsdp_checkpoint(ckpt_dir, device="cpu")   # or "cuda"
#         model.load_state_dict(state_dict, strict=False)
#         model.eval()

#     else:
#         tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
#         tok.pad_token = tok.eos_token
#         model = AutoModelForCausalLM.from_pretrained(args.model_dir,
#                                                      device_map="auto",
#                                                      torch_dtype=torch.bfloat16  # keep dtype consistent with training)
#         )
#         model.eval()

#     # Dataset
#     ds = get_gsm8k(args.split)
#     if args.limit is not None:
#         ds = ds.select(range(min(args.limit, len(ds))))

#     # sampling settings
#     do_sample = args.temperature > 0.0
#     top_p      = 1.0 if do_sample else 1.0

#     # Generation kwargs (same as GRPOConfig defaults)
#     gen_kwargs = dict(
#         do_sample=do_sample,
#         temperature=args.temperature,
#         top_p=top_p,
#         max_new_tokens=args.max_token_len,
        
#         pad_token_id=tok.eos_token_id,
#         return_dict_in_generate=True,
#         num_return_sequences=args.k,
#         stopping_criteria=StoppingCriteriaList([
#             StopOnStrings([
#                 tok.encode("</answer>",    add_special_tokens=False),
#                 tok.encode("<|im_end|>",    add_special_tokens=False),  # Qwen sentinel
#             ])
#         ]),
#     )

#     pass1     = 0      # count of queries where any of k samples is correct
#     total_len = 0      # sum of *all* completion token lengths

#     for i, ex in enumerate(tqdm(ds, desc=f"Evaluating {args.split}"), 1):
#         # build prompt exactly like GRPOTrainer
#         prompt_txt = "".join(m["content"] for m in ex["prompt"])
#         first_device = next(iter(model.hf_device_map.values()))
#         inp = tok(prompt_txt, return_tensors="pt").to(first_device)

#         # replicate prompt k times
#         input_ids     = inp["input_ids"].repeat(args.k, 1)
#         attention_mask = inp["attention_mask"].repeat(args.k, 1)

#         with torch.no_grad():
#             out = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 **gen_kwargs
#             )

#         # strip off prompt to get only completion
#         seqs     = out.sequences  # (k, prompt_len + comp_len)
#         comp_seqs = seqs[:, inp["input_ids"].shape[-1]:]  # (k, comp_len)

#         # update total token count across all k samples
#         total_len += comp_seqs.shape[-1] * args.k

#         # decode & check pass@1
#         preds = [
#             extract_xml_answer(tok.decode(c, skip_special_tokens=False))
#             for c in comp_seqs
#         ]
#         if any(p == ex["answer"] for p in preds):
#             pass1 += 1

#         # running metrics
#         if (i % args.log_interval == 0) or (i == len(ds)):
#             run_pass1   = 100 * pass1 / i
#             run_mean_len = total_len / (i * args.k)
#             print(f"[{i:4d}/{len(ds)}]  "
#                   f"pass@{args.k}={run_pass1:6.2f}%  "
#                   f"mean_len={run_mean_len:6.2f}")

#     # final summary
#     final_pass1   = 100 * pass1 / len(ds)
#     final_mean_len = total_len / (len(ds) * args.k)
#     print("\n=== FINAL RESULTS ===")
#     print(f"pass@{args.k}                 : {pass1}/{len(ds)} = {final_pass1:.2f}%")
#     print(f"Mean completion length : {final_mean_len:.2f} tokens")

# if __name__ == "__main__":
#     main()


# train_grpo.py
#
# See https://github.com/willccbb/verifiers for ongoing developments
#

import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import argparse # 1. ADDED: Import argparse

# --- (All your helper and reward functions remain exactly the same) ---

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
XML_COT_FORMAT = "..." # Truncated for brevity

def extract_xml_answer(text: str) -> str:
    # ... (code is unchanged)
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    # ... (code is unchanged)
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

def get_gsm8k_questions(split = "train") -> Dataset:
    # ... (code is unchanged)
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

# --- (All your reward functions: correctness_reward_func, xmlcount_reward_func, etc. remain exactly the same) ---
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    # ... (code is unchanged)
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    # ... (code is unchanged)
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    # ... (code is unchanged)
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    # ... (code is unchanged)
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def xml_format_reward_func(completions, **kwargs) -> list[float]:
    # ... (code is unchanged)
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    # ... (code is unchanged)
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    # ... (code is unchanged)
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# 2. ADDED: Command-line argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--eval_only", action="store_true", help="Run in evaluation-only mode.")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to the checkpoint to evaluate.")
args = parser.parse_args()


# --- Model and Training Configuration ---
model_name = "Qwen/Qwen2.5-7B-Instruct" # Base model for tokenizer
seed = 42

if args.eval_only:
    # --- EVALUATION MODE ---
    assert args.checkpoint_dir is not None, "Must provide --checkpoint_dir in eval mode."
    dataset = get_gsm8k_questions(split="test")
    model_path = args.checkpoint_dir
    run_name = f"eval-{model_path.split('/')[-1]}" # Create a descriptive run name
    training_args = GRPOConfig(
        output_dir="./eval_logs", # Temporary dir, not used for saving
        run_name=run_name,
        learning_rate=0.0, # The key change: no learning!
        num_train_epochs=1,
        per_device_train_batch_size=1, # Eval one by one
        gradient_accumulation_steps=2,
        save_steps=1_000_000, # Do not save during eval
        logging_steps=1,
        temperature = 1e-16,
        report_to="wandb",
        bf16=True,
        log_on_each_node=False,
        disable_dropout=True,
        # Generation params should match what you want to test
        num_generations=2,
        max_prompt_length=256,
        max_completion_length=786,
    )
else:
    # --- TRAINING MODE ---
    dataset = get_gsm8k_questions(split="train")
    model_path = model_name
    run_name = model_name + '-gsm8k-zero1-' + 'seed' + str(seed)
    output_dir = "outputs/" + run_name
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=1e-6,
        beta = 0.0,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        seed = seed,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=256,
        max_completion_length=786,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
        overwrite_output_dir=True,
        disable_dropout=True,
    )


# 3. MODIFIED: Load model from checkpoint in eval mode
model = AutoModelForCausalLM.from_pretrained(
    model_path, # This is now either the base model or the checkpoint path
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = False

# Tokenizer is always loaded from the base model for consistency
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Initialize the trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
        ],
    args=training_args,
    # Use 'dataset' which is correctly set to train or test split
    train_dataset=dataset,
)

# 4. MODIFIED: The main train/eval call
if args.eval_only:
    print("--- RUNNING IN EVALUATION MODE ---")
    trainer.train() # This will "train" for 1 epoch with LR=0.0, effectively just running inference and logging
else:
    print("--- RUNNING IN TRAINING MODE ---")
    trainer.train()

    # 5. MODIFIED: Final save only happens during training
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        trainer.save_model(output_dir + "/final")
        tokenizer.save_pretrained(output_dir + "/final")
    trainer.accelerator.wait_for_everyone()