# train_grpo.py
#
# See https://github.com/willccbb/verifiers for ongoing developments
#


import re
import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import numpy as np
from accelerate import Accelerator
from functools import partial

# ----------------------------------------------------------------------------
# Argument Parser
# ----------------------------------------------------------------------------
def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a model using GRPOTrainer with configurable arguments.")

    # Model and Tokenizer
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="The name of the base model to use from Hugging Face Hub.")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation (e.g., 'flash_attention_2'). Use 'eager' for non-flash attention.")

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k", help="The name of the dataset to use.")
    parser.add_argument("--dataset_subset", type=str, default="main", help="The subset of the dataset to use.")

    # Training Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="The learning rate for the AdamW optimizer.")
    parser.add_argument("--gamma", type=float, default=1-1e-7, help="The discount factor for controlling lengths of completions in the GRPO loss.")
    parser.add_argument("--beta", type=float, default=0.4, help="The beta parameter for the GRPO loss.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Training batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps for gradient accumulation.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for the learning rate scheduler.")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup", help="Learning rate scheduler type.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for the optimizer.")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Maximum gradient norm for clipping.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature for sampling.")

    # GRPO Specific
    parser.add_argument("--num_generations", type=int, default=4, help="Number of generations per prompt for GRPO.")
    parser.add_argument("--ref_model_sync_steps", type=int, default=32, help="Steps between syncing the reference model.")
    parser.add_argument("--sync_ref_model", action='store_true', help="Disable kl regularization in the model")
    parser.add_argument("--disable_dropout", action='store_true', help="Disable dropout in the model.")


    # Generation Lengths
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Maximum prompt length in tokens.")
    parser.add_argument("--max_completion_length", type=int, default=786, help="Maximum completion length (max_new_tokens).")

    # Run Configuration
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--machine_name", type=str, default='default', help="A name for the machine or environment to append to the run name.")
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for the W&B run. If not set, it's generated automatically.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="The base directory to save model outputs and checkpoints.")
    parser.add_argument("--report_to", type=str, default="wandb", help="Report results to 'wandb' or 'none'.")


    # Logging and Saving
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every N steps.")
    parser.add_argument("--save_steps", type=int, default=400, help="Save a checkpoint every N steps.")

    return parser.parse_args()
args = get_args()




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

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
            #    answer="7"
            #)},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

# Base Reward function (used by others)
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    #print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    # Return a larger reward for correctness to make its signal stronger
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# Other shaping rewards
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def xml_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward 0.1 if the completion contains:
      <reasoning> ... </reasoning>
      <answer>   ... </answer>
    (in that order, allowing any content/newlines in between).
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
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
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# --- Define the necessary reward functions for discounting and evaluation ---

def discounted_correctness_reward(
    prompts, completions, answer, gamma: float, tokenizer, **kwargs
) -> list[float]:
    """Internal function to calculate the fully discounted reward."""
    base_rewards = correctness_reward_func(prompts, completions, answer, **kwargs)
    discounted_rewards = []
    for i, completion in enumerate(completions):
        base_reward = base_rewards[i]
        if base_reward == 0.0:
            discounted_rewards.append(0.0)
            continue
        completion_text = completion[0]['content']
        num_tokens = len(tokenizer.encode(completion_text))
        if num_tokens > 0:
            discounted_reward = base_reward * (gamma ** (num_tokens - 1))
        else:
            discounted_reward = 0.0
        discounted_rewards.append(discounted_reward)
    return discounted_rewards



def training_reward_adjustment(
    prompts, completions, answer, gamma: float, tokenizer, **kwargs
) -> list[float]:
    """
    Calculates the difference for TRAINING: (discounted_reward - undiscounted_reward).
    When added to the undiscounted_reward, the total is the discounted_reward.
    """
    undiscounted_rewards = correctness_reward_func(prompts, completions, answer, **kwargs)
    full_discounted_rewards = discounted_correctness_reward(
        prompts, completions, answer, gamma, tokenizer, **kwargs
    )
    return [d - u for d, u in zip(full_discounted_rewards, undiscounted_rewards)]


# --- Model and Training Configuration ---

#model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = args.model_name
seed=args.seed
machine_name = args.machine_name
GAMMA = args.gamma

model_short_name = args.model_name.split("/")[-1]
run_name = f"{model_short_name}-gsm8k-gamma{GAMMA}-seed{args.seed}-{args.machine_name}"
output_dir = f"{args.output_dir}/{run_name}"


training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=args.learning_rate,
    beta = args.beta,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = args.weight_decay,
    warmup_ratio = args.warmup_ratio,
    lr_scheduler_type=args.lr_scheduler_type,
    logging_steps=args.logging_steps,
    seed = seed,
    bf16=True,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_generations=args.num_generations,
    max_prompt_length=args.max_prompt_length,
    max_completion_length=args.max_completion_length,
    num_train_epochs=args.num_train_epochs,
    save_steps=args.save_steps,
    max_grad_norm=args.max_grad_norm,
    report_to=args.report_to,
    log_on_each_node=False,
    overwrite_output_dir=True,
    disable_dropout=args.disable_dropout,  # Important for consistent generation
    sync_ref_model=args.sync_ref_model,
    ref_model_sync_steps=args.ref_model_sync_steps,
    temperature=args.temperature,
    #ddp_find_unused_parameters=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
#model.gradient_checkpointing_enable()

model.config.use_cache = False


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# --- Configure the GRPOTrainer with the Correct Reward Functions ---


# Create a partial function to pass the gamma and tokenizer to the adjustment function
# This is necessary because the trainer only calls reward functions with a standard set of arguments.
_partial_training_adjustment = partial(
    training_reward_adjustment,
    gamma=GAMMA,
    tokenizer=tokenizer
)

# It's good practice to wrap the partial function to give it a clear __name__ for logging
def training_adjustment_func(prompts, completions, answer, **kwargs):
    """A wrapper to call the partial function, satisfying the GRPOTrainer's __name__ requirement."""
    return _partial_training_adjustment(
        prompts=prompts,
        completions=completions,
        answer=answer,
        **kwargs
    )
training_adjustment_func.__name__ = "training_adjustment_func" # Explicitly set name


# Initialize the trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        # --- Shaping Rewards ---
        # These guide the model towards the correct format and style.
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,

        # --- Correctness Rewards (The Additive Trick) ---
        # The trainer sums all rewards. We provide two functions that, when added,
        # result in the desired discounted reward for training, while allowing us
        # to log the "true" undiscounted reward separately.

        # 1. The Undiscounted Reward (for PLOTTING):
        # This gives the "true" correctness score. The trainer will log this
        # function's output, so you can plot `rewards/undiscounted_correctness_reward/mean`.
        correctness_reward_func,

        # 2. The Adjustment (for TRAINING):
        # This function calculates: (Discounted Reward - Undiscounted Reward).
        # When the trainer adds this to the function above, the undiscounted parts
        # cancel out, leaving only the discounted reward in the final training signal.
        training_adjustment_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

print(trainer.generation_config)  
print("→ max_new_tokens:", trainer.generation_config.max_new_tokens)
print("→ temperature:   ", trainer.generation_config.temperature)
print("→ top_p:         ", trainer.generation_config.top_p)
print("→ gamma:         ", GAMMA)
print("→ ref_sync:      ", args.sync_ref_model)
print("→ disable_dropout:      ", args.disable_dropout)

trainer.train()



# trainer.accelerator.wait_for_everyone()
# if trainer.accelerator.is_main_process:
#     trainer.save_model(output_dir + "/final")   # or your path
#     tokenizer.save_pretrained(output_dir + "/final")
# trainer.accelerator.wait_for_everyone()
