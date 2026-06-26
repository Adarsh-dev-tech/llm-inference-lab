# Theory Reference: Speculative Decoding and Parallel Verification

Speculative decoding (Leviathan et al., 2022) is an optimization method designed to accelerate the autoregressive generation (decode) stage of Large Language Models. This guide provides a comprehensive deconstruction of the mathematical formulations, verification algorithms, and native C++ implementations within `llama.cpp` to serve as a complete reference.

---

## 1. The Bottleneck: Decode Weight Streaming

Autoregressive token generation is **memory-bandwidth bound**. 
*   **The Cause**: At each step $t$, the model processes only a single token ($T=1$). To calculate the next token, the GPU must load every parameter of the model (e.g., $14$ GB of weights for a 7B model at FP16) from VRAM to its execution registers.
*   **The Inefficiency**: The arithmetic intensity is extremely low. The GPU cores sit idle, starved of mathematical operations relative to the massive weight data transfer overhead.
*   **Speculative Decoding Solution**: Use a smaller, cheaper model (the **Draft Model**, e.g., 1.5B) to generate a sequence of $K$ candidate tokens quickly. Then, use the larger model (the **Target Model**, e.g., 7B) to verify all $K$ candidates in a single parallel step. Because prompt evaluation (prefill/verification) processes multiple tokens simultaneously, its arithmetic intensity is high, utilizing the GPU cores efficiently.

---

## 2. Mathematical Formulation of Speculative Verification

To ensure that the output text remains mathematically identical in quality to the target model alone, speculative decoding utilizes a **speculative acceptance rate logic** based on probability distributions.

Let:
*   $q(x)$ be the probability of token $x$ predicted by the Draft Model.
*   $p(x)$ be the probability of token $x$ predicted by the Target Model.

Given $K$ draft tokens generated sequentially by the draft model: $x_1, x_2, \dots, x_K$.

### A. Token-by-Token Verification Loop
For each draft token $x_i$ (where $i = 1 \dots K$):
1.  Compute target model probabilities $p(x_i | x_{<i})$ and draft model probabilities $q(x_i | x_{<i})$.
2.  Sample a random value $U \sim \text{Uniform}(0, 1)$.
3.  **Acceptance Rule**:
    If $U < \min\left(1, \frac{p(x_i)}{q(x_i)}\right)$, the draft token $x_i$ is **accepted**. We proceed to verify the next token $x_{i+1}$.
4.  **Rejection Rule**:
    If $U \ge \min\left(1, \frac{p(x_i)}{q(x_i)}\right)$, the draft token $x_i$ is **rejected**. We discard $x_i$ and all subsequent draft tokens ($x_{i+1 \dots K}$).

### B. Fallback Sampling upon Rejection
If draft token $x_i$ is rejected at step $i$, we sample the replacement token $x_i^*$ from the normalized residual distribution:
$$p^*(x) = \frac{\max\left(0, p(x) - q(x)\right)}{\sum_y \max\left(0, p(y) - q(y)\right)}$$

This recovery distribution ensures that the probability of generating any token matches the target model's original distribution exactly.

### C. Speedup Metric Equation
The theoretical speedup factor $\text{Speedup}$ is modeled as:
$$\text{Speedup} = \frac{1}{\gamma \cdot \alpha + (1 - \alpha) + \beta}$$
Where:
*   $\alpha$: Average draft token acceptance rate (fraction of tokens accepted, typically $0.6 - 0.8$).
*   $\gamma$: Ratio of target model latency to draft model latency (typically $4\text{x} - 8\text{x}$).
*   $\beta$: Relative overhead of parallel verification.

---

## 3. Native C++ Speculative Execution inside llama.cpp

In `llama.cpp` (and the `llama-cli` / `llama-server` tools), speculative decoding is supported natively via specialized context coordination.

### A. Context Synchronization
*   `llama.cpp` initializes two distinct contexts: the **Target Context** (`ctx`) and the **Draft Context** (`ctx_draft`).
*   During execution, the draft context generates up to `--spec-draft-n-max` tokens autoregressively.
*   Once the draft sequence is compiled, their token IDs are appended to the target context's input sequence.
*   `llama_decode` evaluates the entire sequence of draft tokens in the target model in a single parallel batch pass.
*   The C++ engine evaluates the target logits, compares them against the draft probabilities using the acceptance rule, and updates the KV cache positions (removing the entries corresponding to rejected draft tokens).

### B. Command-Line Arguments
*   `-m FNAME`, `--model FNAME`: Path to the large target model GGUF.
*   `-md FNAME`, `--model-draft FNAME`: Path to the small draft model GGUF.
*   `-ngl N`, `--gpu-layers N`: Layers to offload to the GPU for the target model.
*   `-ngld N`, `--gpu-layers-draft N`: Layers to offload to the GPU for the draft model.
*   `--spec-draft-n-max N`: Max draft tokens to guess before verification (default: 3 or 5).
*   `--spec-draft-p-min P`: Minimum acceptance probability threshold.

---

## Related Documentation
*   [Theory Reference: KV Cache Sizing and Memory Layouts](kv-cache.md)
*   [Theory Reference: FlashAttention Fused Kernels and SRAM Tiling](flash-attention.md)
*   [Benchmark Report: Speculative Decoding Speedup Report](../benchmarks/28-06-2026-speculative-decoding.md)
