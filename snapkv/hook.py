import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Any

# Import helper functions from transformers modeling
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv, Qwen2Attention

def snapkv_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Any] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Interceptive forward function for Qwen2Attention to implement SnapKV KV Cache compression.
    Matches the exact signature of Transformers Qwen2Attention.forward.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    bsz = hidden_states.shape[0]
    q_len = hidden_states.shape[1]

    # Query, Key, Value projections
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Apply RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Check if we are in the prefill stage of prompt evaluation
    is_prefill = past_key_values is not None and q_len > 1
    
    num_heads = self.config.num_attention_heads
    num_key_value_heads = self.config.num_key_value_heads
    num_key_value_groups = self.num_key_value_groups
    hidden_size = self.config.hidden_size

    if is_prefill:
        # 1. Compute full attention scores on uncompressed states for prefill stage
        key_states_rep = repeat_kv(key_states, num_key_value_groups)
        value_states_rep = repeat_kv(value_states, num_key_value_groups)
        
        attn_weights = torch.matmul(query_states, key_states_rep.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Prefill output computed on full uncompressed states
        attn_output = torch.matmul(attn_weights, value_states_rep)
        
        # 2. Perform SnapKV clustering and select key indices for caching
        K = getattr(self, "snapkv_k", 64)
        L_obs = min(getattr(self, "snapkv_obs_window", 32), q_len)
        L_rec = min(getattr(self, "snapkv_recent_window", 32), q_len)
        k_len = key_states.shape[-2]
        
        obs_attn = attn_weights[:, :, -L_obs:, :].mean(dim=2)
        obs_attn_grouped = obs_attn.view(bsz, num_key_value_heads, num_key_value_groups, k_len).mean(dim=2)
        prunable_len = k_len - L_rec
        
        if prunable_len > K:
            prunable_attn = obs_attn_grouped[:, :, :prunable_len]
            _, topk_idx = torch.topk(prunable_attn, K, dim=-1)
            recent_idx = torch.arange(prunable_len, k_len, device=hidden_states.device).view(1, 1, -1).expand(bsz, num_key_value_heads, -1)
            selected_idx = torch.cat([topk_idx, recent_idx], dim=-1)
            selected_idx = torch.sort(selected_idx, dim=-1)[0]
            
            # Gather compressed keys and values for storage
            gather_idx = selected_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            compressed_key = torch.gather(key_states, dim=2, index=gather_idx)
            compressed_val = torch.gather(value_states, dim=2, index=gather_idx)
        else:
            compressed_key = key_states
            compressed_val = value_states
            
        # Store compressed states in the cache
        past_key_values.update(compressed_key, compressed_val, self.layer_idx)
    else:
        # Decode stage (single token evaluation) -> normal cache update and attention
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            
        key_states_rep = repeat_kv(key_states, num_key_value_groups)
        value_states_rep = repeat_kv(value_states, num_key_value_groups)
        
        attn_weights = torch.matmul(query_states, key_states_rep.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_rep)

    # Reshape back to original representation
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, hidden_size)

    # Output projections
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def patch_snapkv(model: nn.Module, k: int = 64, obs_window: int = 32, recent_window: int = 32):
    """
    Traverses the model and patches all instances of Qwen2Attention with SnapKV custom forward methods.
    """
    for name, module in model.named_modules():
        if isinstance(module, Qwen2Attention):
            # Bind hyperparameters directly to the module instance
            module.snapkv_k = k
            module.snapkv_obs_window = obs_window
            module.snapkv_recent_window = recent_window
            
            # Monkey-patch the forward method
            # Use __get__ to bind the custom function to the module instance as a method
            module.forward = snapkv_forward.__get__(module, Qwen2Attention)

