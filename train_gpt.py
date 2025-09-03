# train_grpo.py
#
# See https://github.com/willccbb/verifiers for ongoing developments
#

import re
import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/phi-4",
        help="The name of the base model to use from Hugging Face Hub.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="Attention implementation: 'sdpa' (safe default) or 'flash_attention_2' if available.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for generation inside GRPO (faster sampling if installed).",
    )

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k", help="The name of the dataset to use.")
    parser.add_argument("--dataset_subset", type=str, default="main", help="The subset of the dataset to use.")
    parser.add_argument("--shuffle_dataset", action='store_true', help="Shuffle the dataset")

    # Training Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2.5e-6, help="The learning rate for the AdamW optimizer.")
    parser.add_argument("--gamma", type=float, default=1-1e-7, help="The discount factor for controlling lengths of completions in the GRPO loss.")
    parser.add_argument("--beta", type=float, default=0.4, help="The beta parameter for the GRPO loss.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Training batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps for gradient accumulation.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for the learning rate scheduler.")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup", help="Learning rate scheduler type.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for the optimizer.")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Maximum gradient norm for clipping.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature for sampling.")

    # GRPO Specific
    parser.add_argument("--num_generations", type=int, default=4, help="Number of generations per prompt for GRPO.")
    parser.add_argument("--ref_model_sync_steps", type=int, default=32, help="Steps between syncing the reference model.")
    parser.add_argument("--sync_ref_model", action='store_true', help="Disable kl regularization in the model")
    parser.add_argument("--disable_dropout", action='store_true', help="Disable dropout in the model.")

    # Generation Lengths
    parser.add_argument("--max_prompt_length", type=int, default=1536, help="Maximum prompt length in tokens.")
    parser.add_argument("--max_completion_length", type=int, default=3072, help="Maximum completion length (max_new_tokens).")
    parser.add_argument("--max_answer_length", type=int, default=500, help="Maximum answer length for discounted model.")
    parser.add_argument("--answer_reward_scale", type=float, default=1.0, help="Maximum answer length for discounted model.")

    # Run Configuration
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--machine_name", type=str, default='default', help="A name for the machine or environment to append to the run name.")
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for the W&B run. If not set, it's generated automatically.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="The base directory to save model outputs and checkpoints.")
    parser.add_argument("--report_to", type=str, default="wandb", help="Report results to 'wandb' or 'none'.")

    # Logging and Saving
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every N steps.")
    parser.add_argument("--save_steps", type=int, default=6000, help="Save a checkpoint every N steps.")
    return parser.parse_args()

args = get_args()


# Load and prep dataset

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

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""

def get_math_questions(split = "train") -> Dataset:
    data = load_dataset("agentica-org/DeepScaleR-Preview-Dataset")[split]  # type: ignore
    # Keep prompts as chat messages; Phi-4 tokenizer provides a chat template.
    data = data.map(lambda x: {  # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['problem']}
        ],
        'answer': x['answer']
    })  # type: ignore
    return data  # type: ignore

dataset = get_math_questions()
dataset = dataset.shuffle(seed=args.seed).select(range(7000))


# Base Reward function (used by others)
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if verify(parse(r),parse(a)) == True else 0.0 for r, a in zip(extracted_responses, answer)]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def answer_length_reward(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if len(r) < args.max_answer_length and len(r) > 0 and GAMMA * args.answer_reward_scale < 1 else 0.0 for r in extracted_responses]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def xml_format_reward_func(completions, **kwargs) -> list[float]:
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


# --- Discounting helpers ---

RE_REASONING_CONTENT = re.compile(
    r"<reasoning>[^\S\n]*\n(.*?)\n[^\S\n]*</reasoning>",
    re.DOTALL,
)

def discountable_text(content: str, ignore_newlines: bool = True) -> str:
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    parts = RE_REASONING_CONTENT.findall(text)
    if not parts:
        return ""
    reasoning_text = "\n\n".join(parts)
    if ignore_newlines:
        return re.sub(r"\s+", " ", reasoning_text).strip()
    else:
        return reasoning_text

def discounted_correctness_reward(
    prompts, completions, answer, gamma: float, tokenizer, **kwargs
) -> list[float]:
    base_rewards = correctness_reward_func(prompts, completions, answer, **kwargs)
    out = []
    for i, completion in enumerate(completions):
        base = base_rewards[i]
        if base == 0.0:
            out.append(0.0)
            continue
        completion_text = completion[0]["content"]
        countable = discountable_text(completion_text, ignore_newlines=False)
        n_tokens = len(tokenizer.encode(countable))
        k = max(n_tokens, 1)
        out.append(base * (gamma ** (k - 1)))
    return out

def training_reward_adjustment(
    prompts, completions, answer, gamma: float, tokenizer, **kwargs
) -> list[float]:
    undiscounted_rewards = correctness_reward_func(prompts, completions, answer, **kwargs)
    full_discounted_rewards = discounted_correctness_reward(
        prompts, completions, answer, gamma, tokenizer, **kwargs
    )
    return [d - u for d, u in zip(full_discounted_rewards, undiscounted_rewards)]


# --- Model and Training Configuration ---

model_name = args.model_name
seed = args.seed
machine_name = args.machine_name
GAMMA = args.gamma

model_short_name = args.model_name.split("/")[-1]
run_name = f"{model_short_name}-math-gamma{GAMMA}-seed{args.seed}-{args.machine_name}"
output_dir = f"{args.output_dir}/{run_name}"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=args.learning_rate,
    beta=args.beta,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    lr_scheduler_type=args.lr_scheduler_type,
    logging_steps=args.logging_steps,
    seed=seed,
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
    shuffle_dataset=args.shuffle_dataset,
    overwrite_output_dir=True,
    disable_dropout=args.disable_dropout,
    sync_ref_model=args.sync_ref_model,
    ref_model_sync_steps=args.ref_model_sync_steps,
    temperature=args.temperature,
    use_vllm=args.use_vllm,
    generation_kwargs={
        "top_k": -1,
        "temperature": args.temperature,
        "top_p": 1,
        "seed": args.seed,
        "max_tokens": args.max_completion_length,
        "stop": ["</answer>"],
    },
)

# --- Load Phi-4 ---

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation=args.attn_implementation,
    trust_remote_code=True,
)

model.config.use_cache = False  # important for training

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

formatted = tokenizer.apply_chat_template(
    [{'role':'system','content': SYSTEM_PROMPT},
     {'role':'user','content': '2+3?'}],
    add_generation_prompt=True
)
print(formatted[:400])

# Ensure padding is defined (Phi-4 uses chat template with special tokens)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# --- Configure the GRPOTrainer with the Correct Reward Functions ---

_partial_training_adjustment = partial(
    training_reward_adjustment,
    gamma=GAMMA,
    tokenizer=tokenizer
)

def training_adjustment_func(prompts, completions, answer, **kwargs):
    return _partial_training_adjustment(
        prompts=prompts,
        completions=completions,
        answer=answer,
        **kwargs
    )
training_adjustment_func.__name__ = "training_adjustment_func"

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,  # tokenizer holds Phi-4 chat template
    reward_funcs=[
        # Shaping
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        xml_format_reward_func,
        answer_length_reward,

        # Correctness + discounting
        correctness_reward_func,
        training_adjustment_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
# trainer.accelerator.wait_for_everyone()
# if trainer.accelerator.is_main_process:
#     trainer.save_model(output_dir + "/final")
#     tokenizer.save_pretrained(output_dir + "/final")
# trainer.accelerator.wait_for_everyone()
