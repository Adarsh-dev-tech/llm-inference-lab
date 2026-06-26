# Ideas for Future Experiments: Edge LLM Optimization (Month 5 & 6)

Our empirical benchmarks for Speculative Decoding and SnapKV on the RTX 3050 (6GB VRAM) edge GPU have exposed critical hardware bottlenecks. This document compiles ideas and research directions to address these inefficiencies in the upcoming months.

---

## 1. Core Takeaways from Months 1-4 Benchmarks

1. **The Edge GPU Bottleneck**: Speculative decoding fails to yield speedups when the target model is fully on the GPU because of **kernel launch serialization and host scheduling latencies**. In batch size 1, sequential draft kernel executions outweigh the mathematical savings of draft-verification.
2. **Memory Competition**: Standard speculative decoding requires loading a second draft model, which competes for the limited 6GB VRAM. This causes Vulkan context allocation OOM crashes for higher-precision target models.
3. **Pruning Resilience**: SnapKV proves that self-attention is highly sparse; we can prune **81.3%** of the KV cache tokens with almost **0.0% perplexity loss**, provided that positional embeddings (RoPE) are explicitly aligned.

---

## 2. Proposed Research & Prototyping Hypotheses

### Idea A: Self-Speculative Decoding (Draft-Free / Medusa)
* **Hypothesis**: Replacing the separate draft model with parallel, light weight prediction heads attached to the target model's output layer will yield a net speedup under full GPU offloading.
* **Mechanism**: Use multiple parallel feed-forward heads (Medusa heads) that predict future tokens concurrently using the target model's internal hidden states.
* **Why it works on Edge**:
  * Eliminates the need to load a second GGUF model, reclaiming VRAM.
  * Replaces multiple model forward passes with single unified kernel execution steps, reducing host launch latencies.

### Idea B: Entropy-Based Adaptive KV Cache Compaction
* **Hypothesis**: Dynamically adjusting the KV cache compression threshold ($K$) at runtime based on model confidence will improve long-context coherence while keeping average memory footprint low.
* **Mechanism**: 
  * Calculate output token entropy (perplexity) at each decode step.
  * When entropy is low (model is confident), prune aggressively ($K=16$) to save VRAM.
  * When entropy spikes (model is uncertain), dynamically retain more context ($K=128$) to prevent hallucinations.

### Idea C: Mixed-Precision KV Caching
* **Hypothesis**: Quantizing historical prunable key-value states to 4-bit while maintaining the protected local window in FP16 will preserve baseline perplexity while matching 4-bit KV memory savings.
* **Mechanism**: Implement a dynamic cache quantizer that downcasts pruned keys/values to low-bit widths while keeping the last $L_{\text{rec}} = 32$ tokens in Float16 format.
