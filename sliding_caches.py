import torch
from transformers.cache_utils import Cache, CacheLayerMixin



class MySlidingWindowLayer(CacheLayerMixin):
    def __init__(self, sliding_window: int, num_sink_tokens: int = 16, dtype=None, device=None, config=None):
        """
        sliding_window: window size for *query-visible* tokens contributed by the tail,
                        i.e. we store at most (sliding_window - 1) past tokens in the tail.
        num_sink_tokens: number of initial tokens to pin as a never-evicted prefix.
        """
        super().__init__()
        if sliding_window is None or sliding_window < 1:
            raise ValueError("sliding_window must be a positive integer.")
        if num_sink_tokens is None or num_sink_tokens < 0:
            raise ValueError("num_sink_tokens must be >= 0.")

        self.sliding_window   = int(sliding_window)
        self.num_sink_tokens  = int(num_sink_tokens)

        self._dtype  = dtype
        self._device = device
        self.config  = config

        self.is_initialized    = False
        self.cumulative_length = 0

        # Rolling tail (evicting), and persistent sink (never evicted)
        self.keys        = None  # tail K
        self.values      = None  # tail V
        self.sink_keys   = None  # sink K
        self.sink_values = None  # sink V

    # ---- lifecycle ---------------------------------------------------------
    def lazy_initialization(self, key_states: torch.Tensor):
        b, h, _, d = key_states.shape
        dev = self._device or key_states.device
        dt  = self._dtype  or key_states.dtype
        empty = key_states.new_empty((b, h, 0, d), device=dev, dtype=dt)
        # independent empty tensors for each buffer
        self.keys, self.values = empty.clone(), empty.clone()
        self.sink_keys, self.sink_values = empty.clone(), empty.clone()
        self.is_initialized = True

    # ---- main API ----------------------------------------------------------
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_position=None, **_):
        """
        RETURN for this step: full_k, full_v = [sink | tail | ALL new tokens]
        UPDATE for next step:
            - Grow sink with (up to num_sink_tokens - |sink|) leftmost tokens from this new block.
            - Tail becomes last (w-1) tokens of (prev_tail + (new_without_sunk)).
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        B, H, T_new, D = key_states.shape
        self.cumulative_length += T_new

        # How many from the *new* block should join the sink (for NEXT step)?
        sink_len      = self.sink_keys.shape[-2]
        take_to_sink  = max(0, min(self.num_sink_tokens - sink_len, T_new))

        # Split new tokens (only affects NEXT-step storage, not current return)
        ks_sink = key_states[..., :take_to_sink, :]
        vs_sink = value_states[..., :take_to_sink, :]
        ks_rest = key_states[..., take_to_sink:, :]
        vs_rest = value_states[..., take_to_sink:, :]

        # ---- Contract with attention: return [past | ALL new] for THIS step ----
        full_k = torch.cat([self.sink_keys, self.keys, key_states],  dim=-2).contiguous()
        full_v = torch.cat([self.sink_values, self.values, value_states], dim=-2).contiguous()

        # ---- Update internal storage for NEXT step -----------------------------
        # 1) Grow sink up to num_sink_tokens using the earliest part of the new block
        if take_to_sink > 0:
            self.sink_keys   = torch.cat([self.sink_keys,   ks_sink], dim=-2).contiguous()
            self.sink_values = torch.cat([self.sink_values, vs_sink], dim=-2).contiguous()

        # 2) Tail keeps the last (w-1) tokens of (prev_tail + new_without_sunk)
        if self.sliding_window > 1:
            if ks_rest.numel():
                tail_src_k = torch.cat([self.keys,   ks_rest], dim=-2)
                tail_src_v = torch.cat([self.values, vs_rest], dim=-2)
            else:
                tail_src_k, tail_src_v = self.keys, self.values
            keep = self.sliding_window - 1
            self.keys   = tail_src_k[..., -keep:, :].contiguous()
            self.values = tail_src_v[..., -keep:, :].contiguous()
        else:
            # degenerate w=1 -> no tail
            self.keys   = self.keys.new_empty(self.keys.shape[:-2] + (0, self.keys.shape[-1]))
            self.values = self.values.new_empty(self.values.shape[:-2] + (0, self.values.shape[-1]))

        return full_k, full_v

    def get_mask_sizes(self, cache_position: torch.Tensor):
        """
        Build sizes for THIS step's causal mask.
        kv_length must equal the number of keys returned by update():
            len(sink) + len(tail) + qlen
        We return a contiguous prefix [sink|tail|current], so kv_offset = 0.
        """
        qlen     = int(cache_position.shape[0])
        sink_len = 0 if (self.sink_keys is None) else int(self.sink_keys.shape[-2])
        tail_len = 0 if (self.keys      is None) else int(self.keys.shape[-2])
        kv_length = sink_len + tail_len + qlen
        kv_offset = 0
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        # capacity hint only; not strictly enforced by this class
        return self.num_sink_tokens + self.sliding_window

    # ---- beam/search helpers ----------------------------------------------
    def batch_repeat_interleave(self, repeats: int) -> None:
        if not self.is_initialized:
            return
        if self.keys is not None and self.keys.numel():
            self.keys   = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)
        if self.sink_keys is not None and self.sink_keys.numel():
            self.sink_keys   = self.sink_keys.repeat_interleave(repeats, dim=0)
            self.sink_values = self.sink_values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if not self.is_initialized:
            return
        if self.keys is not None and self.keys.numel():
            self.keys   = self.keys.index_select(0, indices)
            self.values = self.values.index_select(0, indices)
        if self.sink_keys is not None and self.sink_keys.numel():
            self.sink_keys   = self.sink_keys.index_select(0, indices)
            self.sink_values = self.sink_values.index_select(0, indices)

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        self.batch_select_indices(beam_idx)

    def crop(self, max_length: int) -> None:
        # Only crop the tail; the sink is persistent by design.
        if self.keys is not None and self.keys.shape[-2] > max_length:
            self.keys   = self.keys[..., :max_length, :].contiguous()
            self.values = self.values[..., :max_length, :].contiguous()

    def to(self, *args, **kwargs):
        if self.keys is not None:
            self.keys   = self.keys.to(*args, **kwargs)
            self.values = self.values.to(*args, **kwargs)
        if self.sink_keys is not None:
            self.sink_keys   = self.sink_keys.to(*args, **kwargs)
            self.sink_values = self.sink_values.to(*args, **kwargs)
        return self

    def as_tuple(self):
        # Represent the state to be reused for the NEXT step: [sink | tail]
        if self.sink_keys is None:
            return (self.keys, self.values)
        if self.keys is None:
            return (self.sink_keys, self.sink_values)
        k = torch.cat([self.sink_keys, self.keys], dim=-2).contiguous()
        v = torch.cat([self.sink_values, self.values], dim=-2).contiguous()
        return (k, v)
