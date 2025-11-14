# sliding_window.py
import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers import AutoTokenizer, AutoModelForCausalLM

def resolve_window(cfg, fallback=4096):
    """
    Pick a sensible sliding-window size from the model config, else fallback.
    Order: cfg.sliding_window -> cfg.max_position_embeddings -> fallback
    """
    win = getattr(cfg, "sliding_window", None)
    if isinstance(win, (list, tuple)) and len(win) > 0:
        win = win[0]
    if win is None or (isinstance(win, int) and win <= 0):
        win = getattr(cfg, "max_position_embeddings", None)
    if win is None or (isinstance(win, int) and win <= 0):
        win = fallback
    return int(win)

class MySlidingWindowLayer(CacheLayerMixin):
    def __init__(self, sliding_window: int, dtype=None, device=None, config=None):
        super().__init__()               # <-- no arguments
        if sliding_window is None:
            raise ValueError("sliding_window is None. Pass a positive integer.")
        self.sliding_window = int(sliding_window)
        self._dtype = dtype
        self._device = device
        self.config = config             # optional; not used by super
        self.is_initialized = False
        self.cumulative_length = 0
        self.keys = None
        self.values = None

    def lazy_initialization(self, key_states: torch.Tensor):
        b, h, _, d = key_states.shape
        dev = self._device or key_states.device
        dt  = self._dtype or key_states.dtype
        empty = key_states.new_empty((b, h, 0, d), device=dev, dtype=dt)
        self.keys, self.values = empty, empty
        self.is_initialized = True

    def update(self, key_states, value_states, cache_position=None, **_):
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        self.cumulative_length += key_states.shape[-2]

        # past + new (returned to attention this step)
        full_k = torch.cat([self.keys,   key_states],  dim=-2)
        full_v = torch.cat([self.values, value_states], dim=-2)

        # retain only last (w-1) for the next step
        if self.sliding_window > 1:
            tail = self.sliding_window - 1
            self.keys   = full_k[..., -tail:, :]
            self.values = full_v[..., -tail:, :]
        else:
            self.keys   = full_k.new_empty(full_k.shape[:-2] + (0, full_k.shape[-1]))
            self.values = full_v.new_empty(full_v.shape[:-2] + (0, full_v.shape[-1]))
        return full_k, full_v

    def get_mask_sizes(self, cache_position):
        q = cache_position.shape[0]
        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        if self.cumulative_length >= self.sliding_window:
            kv_length = (self.sliding_window - 1) + q
        else:
            kv_length = self.cumulative_length + q
        return kv_length, kv_offset

    def get_seq_length(self): return self.cumulative_length
    def get_max_cache_shape(self): return self.sliding_window

    # beam/search helpers
    def batch_repeat_interleave(self, r):
        if self.keys is not None and self.keys.numel():
            self.keys   = self.keys.repeat_interleave(r, dim=0)
            self.values = self.values.repeat_interleave(r, dim=0)

    def batch_select_indices(self, idx):
        if self.keys is not None and self.keys.numel():
            self.keys   = self.keys.index_select(0, idx)
            self.values = self.values.index_select(0, idx)

    def reorder_cache(self, beam_idx): self.batch_select_indices(beam_idx)

    def crop(self, max_length):
        if self.keys is not None and self.keys.shape[-2] > max_length:
            self.keys   = self.keys[..., :max_length, :]
            self.values = self.values[..., :max_length, :]

    def to(self, *a, **k):
        if self.keys is not None:
            self.keys   = self.keys.to(*a, **k)
            self.values = self.values.to(*a, **k)
        return self

    def as_tuple(self): return (self.keys, self.values)


model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained(model_name)

string_to_repeat = "5432465437324556"
inputs = tok("Repeat this string: "+string_to_repeat, return_tensors="pt").to(model.device)

string_len = tok(string_to_repeat, return_tensors="pt").to(model.device)['input_ids'].shape[1] 
print("String length (tokens):", string_len)

win = resolve_window(model.config)      # never None now
layers = [MySlidingWindowLayer(sliding_window=string_len,
                               dtype=model.dtype if hasattr(model, "dtype") else torch.bfloat16,
                               device=model.device if hasattr(model, "device") else None,
                               config=model.config)
          for _ in range(model.config.num_hidden_layers)]

from transformers.cache_utils import Cache
my_cache = Cache(layers=layers)

out_ids = model.generate(**inputs, past_key_values=my_cache, max_new_tokens=20, use_cache=True)
print(out_ids)

prompt_len = inputs["input_ids"].shape[1]

# Take only the new tokens
new_ids = out_ids[:, prompt_len:]                      # (batch, new_len)

# Turn them into strings
new_text = tok.batch_decode(new_ids, skip_special_tokens=True)

for i, txt in enumerate(new_text):
    print(f"[sample {i}] {txt}")