# train_grpo.py
#
# See https://github.com/willccbb/verifiers for ongoing developments
#


import re
import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

from trl import GRPOConfig, GRPOTrainer
import numpy as np
from accelerate import Accelerator
from functools import partial
from math_verify import parse, verify


# ----------------------------------------------------------------------------
# Argument Parser
# ----------------------------------------------------------------------------
def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a model using GRPOTrainer with configurable arguments.")

    # Model and Tokenizer
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="The name of the base model to use from Hugging Face Hub.")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation (e.g., 'flash_attention_2'). Use 'eager' for non-flash attention.")
    parser.add_argument("--window_size", type=int, default=512, help="Sliding window size (tokens) to keep in KV cache (excluding sink).")
    parser.add_argument("--enable_sink_attention", action="store_true", help="Enable sink attention (pins the first S tokens and uses a sliding window for the rest).")
    parser.add_argument("--sink_tokens", type=int, default=16, help="Number of initial tokens to pin in the KV cache.")

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k", help="The name of the dataset to use.")
    parser.add_argument("--dataset_subset", type=str, default="main", help="The subset of the dataset to use.")

    # Training Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="The learning rate for the AdamW optimizer.")
    parser.add_argument("--gamma", type=float, default=1, help="The discount factor for controlling lengths of completions in the GRPO loss.")
    parser.add_argument("--beta", type=float, default=0.1, help="The beta parameter for the GRPO loss.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=6, help="Training batch size per device.")
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
    parser.add_argument("--save_steps", type=int, default=500, help="Save a checkpoint every N steps.")
    return parser.parse_args()
args = get_args()



#model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = args.model_name
seed=args.seed
machine_name = args.machine_name
GAMMA = args.gamma

model_short_name = args.model_name.split("/")[-1]
run_name = f"{model_short_name}-gsm8k-gamma{GAMMA}-seed{args.seed}-{args.machine_name}"
output_dir = f"{args.output_dir}/{run_name}"


# Initialize Weights & Biases
if args.report_to == "wandb":
    wandb.init(project="caching-training", name=run_name) 
    # 'name' will be the run name, 'project' will be the project name


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

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

def extract_xml_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return m.group(1) if m else ""

def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.shuffle(seed=args.seed)  # <-- Add this line
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
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
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    # Return a larger reward for correctness to make its signal stronger
    return [2.0 if verify(parse(r),parse(a)) == True else 0.0 for r, a in zip(extracted_responses, answer)]

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
    return [0.25 if match else 0.0 for match in matches]

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

# Remove entire <answer>...</answer> blocks (ignore their inner text)
RE_ANSWER_BLOCK = re.compile(r"<answer>\s*.*?\s*</answer>", re.IGNORECASE | re.DOTALL)
# Strip XML tags themselves (but keep their inner text if any remains)
RE_XML_TAGS = re.compile(r"</?(?:reasoning|answer)\s*>", re.IGNORECASE)

def discountable_text(content: str, ignore_newlines: bool = True) -> str:
    """
    Text to count for discounting:
      - Drop <answer>...</answer> (and its contents).
      - Remove <reasoning>/<answer> tags themselves.
      - Optionally ignore newlines (default True).
    Everything else remains and is counted.
    """
    # Normalize newlines first
    text = content.replace("\r\n", "\n").replace("\r", "\n")

    # 1) Remove the entire answer block(s)
    text = RE_ANSWER_BLOCK.sub("", text)

    # 2) Remove the tags themselves
    text = RE_XML_TAGS.sub("", text)

    # 3) Ignore newlines for discounting
    if ignore_newlines:
        # Replace runs of optional spaces + newline + optional spaces with a single space
        text = re.sub(r"\s*\n\s*", " ", text)
        # Clean up any excessive spacing at the ends
        text = text.strip()

    return text

def discounted_correctness_reward(
    prompts, completions, answer, gamma: float, tokenizer, **kwargs
) -> list[float]:
    """
    Discount using tokens from everything outside <answer>...</answer>,
    excluding the XML tags themselves and (by default) NEWLINES.
    """
    base_rewards = correctness_reward_func(prompts, completions, answer, **kwargs)
    out = []

    for i, completion in enumerate(completions):
        base = base_rewards[i]
        if base == 0.0:
            out.append(0.0)
            continue

        completion_text = completion[0]["content"]

        # Get only the discountable text per your rules
        countable = discountable_text(completion_text, ignore_newlines=True)

        # Tokenize and compute discount
        n_tokens = len(tokenizer.encode(countable))

        # If nothing to count (e.g., model only returned a clean <answer> block),
        # don't penalize formatting: exponent 0 -> gamma**0 == 1
        k = max(n_tokens, 1)
        out.append(base * (gamma ** (k - 1)))

    return out



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



gen_kwargs = {
    "do_sample": True,
    "temperature": args.temperature,
    "top_p": 1.0,
    #"top_k": -1, # Not sure why this does not work anymore
    "max_new_tokens": args.max_completion_length
}

# Add sliding window / sink attention if specified
if args.enable_sink_attention:
    gen_kwargs.update({
        "custom_generate": "transformers-community/sink_cache",
        "trust_remote_code": True,
        "window_length": args.window_size,
        "num_sink_tokens": args.sink_tokens,
    })


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
    #cache_implementation="sliding_window"
    #use_vllm=False,
    # Wire up HF generation kwargs (TRL will pass these to model.generate)
    #generation_kwargs=gen_kwargs,
    #ddp_find_unused_parameters=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation=args.attn_implementation
)
#model.gradient_checkpointing_enable()

model.config.use_cache = True  # training forward passes (GRPO) stay cache-free

# 1) Choose k (defaults if flag is missing)
last_k = args.window_size or 512

# 2) Configure model to use a sliding-window cache at generation time
#    (generate() will instantiate a SlidingWindowCache for you)
model.generation_config.use_cache = True

# 3) Tell the model how long the window should be
#    Many models (incl. Qwen2.x) read this from config during attention masking
model.config.sliding_window = last_k


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"              
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

#### 
from transformers.cache_utils import Cache, CacheLayerMixin

from sliding_caches import MySlidingWindowLayer2

from transformers.cache_utils import Cache

def resolve_window(cfg, fallback=512):
    # prefer CLI --window_size, else config.sliding_window/max_position_embeddings, else fallback
    win = args.window_size or getattr(cfg, "sliding_window", None) or getattr(cfg, "max_position_embeddings", None) or fallback
    if isinstance(win, (list, tuple)):  # some configs store [window] or [left,right]
        win = win[0]
    return int(win)

WIN = resolve_window(model.config)
WIN = args.window_size

SINK = args.sink_tokens
print("Running with window size:" + str(WIN)+" and sink tokens: "+str(SINK))
def make_my_cache(model, win, sink):
    layers = [
        MySlidingWindowLayer(
            sliding_window=win,
            num_sink_tokens=sink,
            dtype=getattr(model, "dtype", torch.bfloat16),
            device=None,
            config=model.config,
        )
        for _ in range(model.config.num_hidden_layers)
    ]
    return Cache(layers=layers)
# Patch model.generate so GRPOTrainer will automatically use your cache.
_orig_generate = model.generate
def _generate_with_my_cache(*args, **kwargs):
    # if the caller didn't supply a cache, inject a fresh one
    kwargs.setdefault("past_key_values", make_my_cache(model, WIN, SINK))
    kwargs.setdefault("use_cache", True)
    return _orig_generate(*args, **kwargs)

model.generate = _generate_with_my_cache

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
        xml_format_reward_func,
        

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



trainer.train()

# trainer.accelerator.wait_for_everyone()
# if trainer.accelerator.is_main_process:
#     trainer.save_model(output_dir + "/final")   # or your path
#     tokenizer.save_pretrained(output_dir + "/final")
# trainer.accelerator.wait_for_everyone()


