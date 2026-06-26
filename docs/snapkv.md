# Theory Reference: SnapKV KV Cache Compression and Attention Sparsity

SnapKV (Li et al., 2024) is a lossy Key-Value (KV) cache compression algorithm designed to mitigate the memory footprint of Large Language Models (LLMs) during long-context generation. By exploiting **attention sparsity**, SnapKV selectively prunes the KV cache during the prompt evaluation (prefill) stage, retaining only the most critical states.

---

## 1. The Core Problem: The KV Cache Memory Wall

In autoregressive decoding, the self-attention layer retrieves the key and value states of all past tokens. As the sequence length $T$ grows, the KV cache footprint scales linearly ($O(T)$):

$$\text{KV Cache Size (Bytes)} = 2 \times L \times H_{\text{kv}} \times D \times \text{BPE} \times T \times B$$

On consumer GPUs (such as a 6GB VRAM card), this creates a **memory wall**:
* A 7B model loaded in 4-bit precision consumes $\approx 5.3\text{ GB}$ of VRAM.
* A standard $32\text{K}$ context window in Float16 requires $\approx 1.8\text{ GB}$ of KV cache space.
* This combination causes immediate Out-of-Memory (OOM) failures or triggers slow CPU memory swapping.

---

## 2. Attention Sparsity and "Heavy Hitters"

Empirical analysis of LLM self-attention matrices shows that attention is highly sparse. When generating new text, the model does not attend to all historical tokens equally. Instead, attention weights are heavily concentrated on:
1. **Local Context**: The most recent tokens (representing immediate syntax, grammar, and active sentence structures).
2. **Heavy Hitters**: A small, persistent set of historical tokens that represent core subject nouns, paragraph transitions, or initiation symbols (like the BOS token).

The vast majority of historical tokens receive virtually $0.0\%$ attention weight during decoding, meaning storing their KV states is highly inefficient.

---

## 3. The SnapKV Algorithm Breakdown

SnapKV identifies and prunes these redundant KV states during the **prefill stage** (prompt evaluation) by analyzing the model's self-attention patterns. It structures key-value selection using three distinct windows:

```
                  Prompt Tokens (Sequence Length T)
[========================================================================]
  \________________________/ \____________________/ \__________________/
        Prunable Pool          Observation Window       Recent Window
     (Select Top-K Keys)       (L_obs query tokens)  (L_rec tokens, kept)
```

### Step 1: The Observation Window ($L_{\text{obs}}$)
To identify which keys are globally important, SnapKV inspects the attention weights of the last $L_{\text{obs}}$ query tokens in the prompt (usually $L_{\text{obs}} = 32$). The hypothesis is that the tokens at the very end of the prompt represent the immediate context leading into text generation, and the historical keys they attend to are highly likely to be the same keys the model will attend to during decoding.

For a query token $q_i$ at index $i$ and a key token $k_j$ at index $j$, the raw attention score is $S_{i, j}$. The softmax attention weights $A_{i, j}$ are averaged across the observation queries:

$$\text{Importance}(j) = \frac{1}{L_{\text{obs}}} \sum_{i = T - L_{\text{obs}}}^{T - 1} A_{i, j}$$

### Step 2: The Recent Window ($L_{\text{rec}}$)
The last $L_{\text{rec}}$ tokens of the prompt (usually $L_{\text{rec}} = 32$) are **always protected and retained** in the cache, bypassing the selection logic. This preserves short-term memory (local context) necessary for immediate fluency.

### Step 3: Top-$K$ Selection and Chronological Sorting
* From the prunable pool (tokens $0$ to $T - L_{\text{rec}} - 1$), the algorithm extracts the indices of the keys with the top $K$ highest importance scores.
* These top-$K$ indices are concatenated with the indices of the protected recent window:
  $$\text{selected\_idx} = \text{TopK}(\text{Importance}_{0 \dots T - L_{\text{rec}} - 1}, K) \cup \{T - L_{\text{rec}}, \dots, T - 1\}$$
* The indices are **sorted chronologically** in ascending order. Chronological sorting is crucial to preserve relative position relationships when applying Rotary Position Embeddings.
* The key and value tensors are gathered from the original tensors using these sorted indices.

---

## 4. Grouped-Query Attention (GQA) Pooling

Modern architectures (like Qwen2.5) use Grouped-Query Attention, where $G$ Query heads share a single Key-Value head. To compress GQA states, SnapKV aggregates attention weights across query heads within each group:

1. **Query-to-KV Grouping**: The attention weights tensor of shape `[batch_size, num_heads, q_len, k_len]` is averaged across the observation window queries, yielding a tensor of shape `[batch_size, num_heads, k_len]`.
2. **Mean Pooling**: Since $G$ query heads share 1 KV head, the attention weights are reshaped to `[batch_size, num_kv_heads, G, k_len]`. We take the mean across the group dimension $G$ to obtain a pooled importance score matching the KV head structure:
   $$\text{PooledScore}(h_{\text{kv}}, j) = \frac{1}{G} \sum_{g=1}^{G} \text{Importance}(h_{\text{kv}} \cdot G + g, j)$$
3. **Selection**: Top-$K$ keys are selected using the `PooledScore`, ensuring that the pruned cache meets the collective attention requirements of all query heads in that group.

---

## 5. The Critical Importance of Position ID Alignment

A major implementation challenge with compressed caches is the calculation of **Rotary Position Embeddings (RoPE)** during autoregressive decoding.

### The Problem
* RoPE relies on the absolute position of a token to rotate its query and key representations.
* When the KV cache is pruned from length $T$ (e.g., 512) to $K_{\text{total}}$ (e.g., 96), standard model runners will query the length of the cache (`past_key_values.get_seq_length()`) to infer the next token's position ID.
* The model incorrectly assigns the first decode step token position ID as $K_{\text{total}}$ (96) instead of the true sequence index $T$ (512).
* This misaligns the query's RoPE rotation relative to the cached keys (which were rotated based on their original prompt indices), causing self-attention to fail completely (catastrophic perplexity spike).

### The Fix
To support SnapKV, the model evaluator must bypass cache-length inference and pass explicit `position_ids` during generation:
* **Prefill pass**: `position_ids = [0, 1, ..., T-1]`.
* **Decode step $i$**: `position_ids = [T + i]`.
This guarantees that query tokens are rotated with their true absolute positional values, maintaining correct relative distances to the keys stored in the compressed cache.

---

## Related Documentation
*   [Learning Report: SnapKV Mechanics and Sparsity](../learnings/learning-snapkv-mechanics.md)
*   [Theory Reference: KV Cache Sizing and Memory Layouts](kv-cache.md)
*   [Benchmark Report: SnapKV Performance and Perplexity Sweeps](../benchmarks/26-06-2026-snapkv-compression.md)
