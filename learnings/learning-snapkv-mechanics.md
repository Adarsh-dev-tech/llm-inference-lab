# Learning Report: SnapKV Mechanics, KV Cache Compression, and Attention Sparsity

This report analyzes the system mechanics of **SnapKV** (Li et al., 2024), an algorithm for dynamic Key-Value (KV) cache compression. We deconstruct how attention sparsity allows for cache pruning, the roles of the observation and recent windows, and how KV pooling maps to Grouped-Query Attention (GQA).

---

## 1. The Phenomenon of Attention Sparsity

Large Language Models (LLMs) are often deployed with massive context limits (e.g., 32K tokens). However, empirical analysis of self-attention matrices reveals that **attention is highly sparse**.
*   **The Observation**: During the autoregressive decoding phase, the model does not attend to all historical tokens equally. Instead, attention weights are heavily concentrated on a tiny subset of "heavy hitters" (semantic key-points, sentence initiators, punctuation) and the most recent local tokens.
*   **The Inefficiency**: Standard cache managers statically allocate VRAM for *every single token* in the history, even if the model has a 0.0% probability of ever attending to that token again.
*   **SnapKV's Objective**: Identify and retain only the critical key-value states, discarding the irrelevant slots to compress VRAM footprint without losing context awareness.

---

## 2. The SnapKV Algorithm Breakdown

SnapKV prunes the KV cache during the prompt evaluation (**prefill**) stage. It structures key selection using three windows:

```
  Prompt Tokens (Sequence Length T)
[========================================================================]
  \________________________/ \____________________/ \__________________/
        Prunable Pool          Observation Window       Recent Window
     (Select Top-K Keys)       (L_obs query tokens)  (L_rec tokens, kept)
```

### A. The Observation Window ($L_{\text{obs}}$)
*   To determine which historical keys are important for the future generation, SnapKV inspects the attention weights of the last $L_{\text{obs}}$ tokens in the prompt (typically the last $32$ tokens).
*   **Logic**: The tokens at the very end of the prompt represent the immediate context leading into text generation. The historical tokens that these final prompt tokens attend to are highly likely to be the same tokens the model will attend to during subsequent decoding steps.

### B. The Recent/Local Window ($L_{\text{rec}}$)
*   The last $L_{\text{rec}}$ tokens of the prompt (typically the last $32$ to $64$ tokens) are **always retained** in the cache, bypassing the pruning logic entirely.
*   **Logic**: Local context is critical for grammar, syntax, and sentence coherence. Preserving the local window intact ensures that the model does not lose its immediate short-term memory.

### C. The Top-$K$ Key Selection
*   For the historical tokens outside the recent window (from token $0$ to $T - L_{\text{rec}} - 1$), SnapKV averages the attention weights across the observation window:
    $$\text{Importance}(j) = \frac{1}{L_{\text{obs}}} \sum_{i = T - L_{\text{obs}}}^{T - 1} A_{i, j}$$
*   The keys with the top $K$ highest importance scores are selected.
*   The final compressed KV cache is constructed by concatenating these top $K$ key-value pairs with the protected local window, sorted in their original chronological order (critical for relative positional embeddings like RoPE).

---

## 3. KV Pooling with Grouped-Query Attention (GQA)

Modern models (like Qwen2.5) use Grouped-Query Attention, where $G$ Query heads share a single Key-Value head. SnapKV adapts to GQA by pooling attention weights:
1.  **Group Aggregation**: The attention weights of shape `[batch_size, num_heads, k_len]` are grouped into `[batch_size, num_kv_heads, G, k_len]`.
2.  **Mean Pooling**: We take the mean across the group dimension ($G$) to obtain a pooled importance score of shape `[batch_size, num_kv_heads, k_len]` matching the KV head dimensions:
    $$\text{PooledScore}(h_{\text{kv}}, j) = \frac{1}{G} \sum_{g=1}^{G} \text{Importance}(h_{\text{kv}} \cdot G + g, j)$$
3.  **Selection**: The top $K$ keys are selected using these grouped scores, ensuring that the selected slots accommodate the shared needs of all Query heads in each group.

---

## 4. SnapKV vs. Alternative Compression Algorithms

| Algorithm | Pruning Stage | Selection Criteria | Positional Consistency | VRAM Saving |
| :--- | :--- | :--- | :--- | :--- |
| **Heavy Hitter Oracle (H2O)** | **Decode** (Step-by-step) | Cumulative attention scores | Dynamic (keeps updating) | High (dynamic evictions) |
| **Static Eviction** | **Decode** (Step-by-step) | FIFO (Discard oldest tokens) | Drop oldest | High (but breaks long context) |
| **SnapKV** | **Prefill** (Single pass) | Observation window pooling | Fixed after prefill (preserves chronology) | High (up to 10x cache compression) |

---

## 5. Performance Trade-offs: Perplexity vs. $K$

The retention limit $K$ controls the trade-off between VRAM footprint and model accuracy (measured by **Perplexity / PPL**):
*   **Large $K$ (e.g. $K=256$)**: The compressed cache size is larger, but perplexity remains almost identical to the uncompressed FP16 baseline. The model retains high generation quality.
*   **Small $K$ (e.g. $K=16$)**: The cache is compressed aggressively, reclaiming massive amounts of VRAM. However, discarding too many historical keys introduces quantization-like noise and attention loss, causing perplexity to rise (leading to repetitive loops or gibberish output).
*   **Optimal Threshold**: Empirical sweeps demonstrate that $K \in \{64, 128\}$ represents the "sweet spot" where the model achieves **5x to 8x VRAM savings** with near-zero perplexity degradation.

---

## Related Documentation
*   [Theory Reference: KV Cache Sizing and Memory Layouts](../docs/kv-cache.md)
*   [Theory Reference: FlashAttention Fused Kernels and SRAM Tiling](../docs/flash-attention.md)
*   [Theory Reference: SnapKV Compression and Attention Sparsity](../docs/snapkv.md)
*   [Benchmark Report: SnapKV Compression and Perplexity Sweeps](../benchmarks/26-06-2026-snapkv-compression.md)
