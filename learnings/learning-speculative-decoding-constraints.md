# Learning Report: Speculative Decoding Constraints and Consumer VRAM Allocation Layouts

This report explores the architectural and resource constraints of running **Speculative Decoding** on consumer graphics cards (specifically target 6GB VRAM GPUs). We analyze dual-model capacity constraints, bandwidth constraints during draft-target coordination, and the math governing throughput speedups.

---

## 1. The Dual-Model Memory Challenge

Speculative decoding requires loading two distinct model networks into memory concurrently:
1.  **The Target Model ($M_t$)**: The primary, large language model (e.g., `Qwen2.5-7B-Instruct` in `Q4_K_M` GGUF, requiring ~4.7 GB).
2.  **The Draft Model ($M_d$)**: The smaller, faster helper model (e.g., `Qwen2.5-1.5B-Instruct` in `Q4_K_M` GGUF, requiring ~1.2 GB).

### VRAM Budget Equation
The total VRAM requirements are defined by:
$$\text{VRAM}_{\text{total}} = \text{Model}_t + \text{Model}_d + \text{Context}_t + \text{Context}_d + \text{Activation Memory}$$

For our RTX 3050 Laptop (6GB VRAM = 6144 MB):
*   $M_t$ (Q4_K_M 7B) footprint: **~4700 MB**
*   $M_d$ (Q4_K_M 1.5B) footprint: **~1200 MB**
*   **Static weight footprint**: $4700 + 1200 = \mathbf{5900\text{ MB}}$

This leaves only **~244 MB** for prompt activations, KV context buffers, and system driver overhead. This budget is dangerously close to the physical limit.

---

## 2. Hardware Allocation and Layer Partitioning

To avoid VRAM spillage and virtual memory swapping, we must divide layer offloading (`-ngl` / `--gpu-layers` and `-ngld` / `--gpu-layers-draft`) strategically between the CPU and GPU.

```
       [GPU VRAM (6GB)]                 [System RAM]
+------------------------------+  +----------------------+
| Target Model (Layers 0-18)   |  | Target (Layers 19-28)|
| Draft Model (Layers 0-28)    |  |                      |
| KV Caches & Activations      |  |                      |
+------------------------------+  +----------------------+
```

### Allocation Strategies:
1.  **Full GPU Draft, Partial GPU Target (Recommended)**:
    Since the draft model runs autoregressively ($K$ steps), its latency directly impacts overall speed. We offload 100% of the draft model layers to the GPU to maximize its tokens-per-second (TPS), and offload only the remaining target layers to the GPU, leaving the overflow layers on the CPU.
2.  **Symmetric Partial Offloading**:
    Dividing GPU layers equally between both models can cause both to run slower because of context switching overhead within the GPU scheduler.

---

## 3. The Latency Trade-Off Math

The actual throughput speedup depends on the **draft model acceptance rate** ($\alpha$) and the **relative latency ratio** ($\gamma$).

Let:
*   $T_t$: Time for the Target Model to decode a single token (~35ms).
*   $T_d$: Time for the Draft Model to decode a single token (~8ms).
*   $K$: Number of draft tokens generated per verification block (speculative window size, e.g., $K=4$).
*   $\alpha$: Probability of a draft token being accepted (typically $0.6 - 0.75$).

### The Verification Latency Equation
The average cost to verify and generate a block of tokens is:
$$\text{Latency}_{\text{block}} = K \cdot T_d + T_t$$
Since the target model evaluates all $K$ tokens in parallel, the verification pass takes approximately the same time as a single autoregressive step ($T_t$).

The average number of accepted tokens per block is:
$$\text{Tokens}_{\text{accepted}} = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$
For $\alpha = 0.7$ and $K = 4$, the average tokens generated is **~3.1 tokens**.

### Speedup Factor:
$$\text{Speedup} = \frac{\text{Tokens}_{\text{accepted}} \cdot T_t}{K \cdot T_d + T_t}$$
*   **Optimal Case**: High acceptance rate ($\alpha \ge 0.75$) leads to **1.5x - 2.2x speedups**.
*   **Pathological Case**: If $\alpha < 0.3$ (poor semantic alignment between draft and target), the speedup drops below **0.8x** (slower than target-only decoding) due to draft generation overhead.

---

## Related Documentation
*   [Theory Reference: Speculative Decoding and Parallel Verification](../docs/speculative-decoding.md)
*   [Benchmark Report: Speculative Decoding Performance Sweeps](../benchmarks/28-06-2026-speculative-decoding.md)
