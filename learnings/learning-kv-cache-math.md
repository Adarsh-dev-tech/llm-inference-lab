# Learning Report: KV Cache Math, Arithmetic Intensity, and Hardware Limits

This learning report explores the mathematical and architectural limits of Key-Value (KV) caching on consumer hardware, focusing on the systems-level bottlenecks of prefill vs. decode stages, the role of Grouped-Query Attention (GQA), and strategies to bypass memory scaling constraints.

---

## 1. Why GQA is a Game-Changer for Memory

In a traditional Multi-Head Attention (MHA) model, every attention head has a dedicated Key and Value head. As context scaling became a priority, MHA created an unsustainable VRAM bottleneck.
*   **The Math**: By sharing a single Key-Value head group across multiple Query heads (Grouped-Query Attention), we divide the KV cache parameters by the grouping factor.
*   **Qwen2.5-7B Case Study**:
    *   Query heads ($H_q$) = $28$
    *   Key-Value heads ($H_{kv}$) = $4$ (Group group size = $7$)
    *   At a context length of $4,096$ tokens, GQA stores $4$ KV vectors instead of $28$ per layer. This compresses the KV cache footprint from **$1.64$ GB** down to **$234.9$ MB** (a **$7.0$x reduction**).
*   **Systems Impact**: By shrinking the KV cache footprint, GQA frees up massive portions of GPU VRAM. This enables consumer-grade GPUs (such as 6 GB cards) to run models up to $16,384$ tokens without running out of memory.

---

## 2. Prefill vs. Decode: The Bottleneck Shift

Profiling LLM execution reveals a fundamental shift in hardware bottlenecks between the two main inference stages: **Prefill** and **Decode**. This shift is explained by **Arithmetic Intensity** (the ratio of floating-point operations performed per byte of memory transferred).

### A. The Prefill Stage (Compute-Bound)
During prefill, the model processes the entire input prompt (e.g., $1,024$ tokens) in a single pass.
*   **Arithmetic Intensity**: **High**. Because we process $T$ tokens simultaneously, the projection weights are multiplied by a large batch matrix $[T, C]$. The operations scale as $O(T \cdot C^2)$, performing thousands of FLOPs for every byte of weight data loaded into the GPU registers.
*   **System Bottleneck**: **Compute-Bound** (saturates GPU CUDA cores).
*   **Observation**: Prefill is highly parallelized. Prompt processing speed is fast (e.g., Qwen2.5-7B runs prefill at over $700$ tokens/sec on an RTX 3050).

### B. The Decode Stage (Memory-Bandwidth-Bound)
During decode, the model generates only *one single token* per step ($T=1$).
*   **Arithmetic Intensity**: **Extremely Low**. For each layer, the GPU must load **all** model weights from VRAM to multiply them by the single new token vector. Since $T=1$, we do $O(C^2)$ operations. We do very little calculation relative to the massive amount of parameter data transferred.
*   **System Bottleneck**: **Memory-Bandwidth-Bound** (saturates VRAM read/write bandwidth).
*   **Observation**: The speed of decoding is directly limited by how fast the GPU can stream weights from VRAM to its execution registers. For a 7B model at FP16 ($\approx 14$ GB of weights), we would need a memory bandwidth of $420$ GB/s just to generate $30$ tokens per second!

---

## 3. The 6 GB VRAM Cliff & Swapping Dynamics

Our benchmarks on the 6 GB RTX 3050 Laptop GPU demonstrated the physical consequences of the memory-bandwidth bottleneck:

### A. The Cost of PCIe Swapping
When loading the **Q8_0** model at full offload (`ngl=33`), the weights ($7.7$ GB) exceed the physical VRAM ($6.0$ GB). The OS pages the remaining $1.7$ GB to system RAM over the PCIe bus.
*   At every decoding step, the GPU must stream this paged weight segment over the PCIe Gen 4 x4 lane (which maxes out at a theoretical $\approx 7.8$ GB/s, compared to local VRAM bandwidth of $\approx 192$ GB/s).
*   This drops generation speed from a local offloaded speed of $\approx 30$ t/s to **$0.46$ t/s** (a **$65$x slowdown**).

### B. The Fallback Solution: Partial CPU Offloading
By reducing the offload target to `ngl=15` layers:
*   We keep $15$ layers strictly in local VRAM ($4.1$ GB) and route the remaining $13$ layers to the CPU.
*   Since the CPU processes its 13 layers using system RAM directly, there is no dynamic, high-frequency weight paging over the PCIe bus during execution.
*   **Result**: Speed reaches **$8.22$ t/s** (an **18x improvement** over the paging-riddled full offload).
*   **Lesson**: In memory-constrained systems, it is always better to offload only what fits cleanly inside physical VRAM, allowing the CPU to execute the rest, rather than over-allocating VRAM and triggering OS paging.

---

## 4. Advanced Strategies to Optimize KV Cache Overhead

As context lengths scale to $32$K or $128$K tokens, even GQA KV caches will saturate VRAM. Three primary technologies mitigate this scaling ceiling:

1.  **PagedAttention (vLLM)**:
    *   Standard llama.cpp allocates contiguous blocks of VRAM for the KV cache. This leads to heavy memory fragmentation (similar to standard OS heap allocation).
    *   PagedAttention divides the KV cache into small, non-contiguous "pages" in VRAM, mapping them dynamically via a page table. This eliminates memory fragmentation and reclaims up to $96\%$ of wasted cache space.
2.  **FlashAttention**:
    *   Standard attention writes the intermediate $T \times T$ attention matrix back to high-bandwidth VRAM, then reads it back to multiply it by the Value matrix.
    *   FlashAttention uses tiled online softmax to compute attention incrementally inside the GPU's ultra-fast, local **SRAM** registers, bypassing slow VRAM reads/writes completely. This dramatically reduces memory traffic during the prefill phase.
3.  **KV Cache Quantization (8-bit / 4-bit cache)**:
    *   Quantizing the Key-Value entries from Float16 to 8-bit or 4-bit formats directly halves or quarters the cache size. This frees up VRAM and speeds up decode by reducing the volume of historical cache data that must be loaded from VRAM at each step.

---

## Related Documentation
*   [Theory Reference: KV Cache Sizing and Memory Layouts](../docs/kv-cache.md)
*   [Benchmark Report: KV Cache Memory Profile on Qwen models](../benchmarks/23-06-2026-kv-cache-profiler.md)
