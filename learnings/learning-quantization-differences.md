# Learning Report: Quantization Trade-Offs and Memory Bandwidth Limits

## Overview
This report explains the critical trade-offs between model weight quantization bit-widths, memory bandwidth ceilings, and reasoning capabilities (perplexity) on consumer hardware, using findings from our Qwen2.5-7B sweeps on the NVIDIA RTX 3050 Laptop GPU (6 GB VRAM).

---

## 1. Why Q4_K_M is the Optimal Default for a 6 GB VRAM Target

On resource-constrained hardware with a strict 6 GB VRAM limit, **Q4_K_M** is the most robust and stable default choice. This is due to two system-level factors:

### Weight Size vs. VRAM Allocation Headroom
*   **Q4_K_M Weight Footprint**: ~4.80 GB. When loaded into VRAM, it leaves **~1.20 GB of free space** (VRAM headroom).
*   **Q5_K_M Weight Footprint**: ~5.43 GB. When loaded, it leaves only **~0.57 GB of free space**.
*   **Q8_0 Weight Footprint**: ~8.00 GB. This exceeds the VRAM limit entirely, forcing host memory paging and destroying execution speed.

### The KV Cache Headroom Safeguard
VRAM is occupied by both static model weights and the **dynamic Key-Value (KV) cache**. The KV cache stores attention vectors for all tokens in the context window and scales linearly with batch size and context length:
$$\text{KV Cache Size (bytes)} = 2 \times \text{layers} \times \text{heads} \times \text{head\_dim} \times \text{bytes\_per\_element} \times \text{context\_len} \times \text{batch\_size}$$

*   For `Qwen2.5-7B` at FP16, a context length of **2048 tokens** and a batch size of **1** requires **~0.25 GB** of VRAM for the KV cache.
*   If using **Q5_K_M**, the static weights consume ~5.58 GB (including baseline OS overhead). Adding a 2048-token context cache (~0.25 GB) pushes total VRAM usage to ~5.83 GB—dangerously close to the 6.0 GB physical ceiling. Any increase in context length (e.g. up to 4096 tokens) or batch size will immediately trigger VRAM overflow.
*   If using **Q4_K_M**, the extra ~0.60 GB of VRAM headroom acts as a buffer. This safeguard allows the KV cache to expand dynamically for longer conversations or larger batches without overflowing the VRAM limit.

---

## 2. Quantization Bit-Width Trade-Offs

Choosing a quantization level involves balancing three variables:

```
                  +-----------------------------------------+
                  |         The Quantization Trilemma       |
                  +-----------------------------------------+
                                       /\
                                      /  \
                                     /    \
                                    /      \
                      [Speed]      /________\     [Perplexity]
                     (Throughput)                  (Reasoning)
                                       \  /
                                        \/
                                     [Memory]
                                    (VRAM/RAM)
```

1.  **VRAM Footprint (Memory)**: Lower bit-widths compress weights, reducing storage size.
2.  **Throughput (Speed)**: In memory-bandwidth-bound environments, loading smaller weights accelerates execution because fewer bytes cross the memory bus.
3.  **Perplexity (Reasoning Quality)**: Higher quantization compression (fewer bits per weight) removes subtle parameters, which can cause accuracy loss (higher perplexity), repetitive phrase loops, or reasoning degradation.

| Quantization Format | Weight Footprint | Average Decode Speed | Perplexity Impact | VRAM Safety Level |
| :--- | :--- | :--- | :--- | :--- |
| **Q4_K_M** (Mixed 4-bit) | ~4.80 GB | **33.63 t/s** | Slight increase | **High** (Safe for context scaling) |
| **Q5_K_M** (Mixed 5-bit) | ~5.43 GB | **29.30 t/s** | Minimal increase | **Moderate** (Risk of OOM at high context) |
| **Q8_0** (Standard 8-bit) | ~8.00 GB | **4.08 t/s** | Near-zero loss | **Unusable** (Triggers VRAM cliff) |

---

## 3. Systems Analysis: Memory Bandwidth Limits and Driver Swapping

The performance data collected reveals two core systems-engineering insights:

### The VRAM Cliff & Driver Swapping
When the model weights exceed the physical GPU memory capacity (as observed with Q8_0's ~8.00 GB footprint on our 6 GB card), the CUDA driver (WDDM on Windows) begins paging memory. Tensors are dynamically moved back and forth between host RAM and GPU VRAM over the PCIe bus during every forward pass. 
*   Because the PCIe Gen4 x8 interface (max ~16 GB/s transfer rate) is much slower than the GPU's internal memory bus, execution units stall.
*   Throughput drops from a fast **29.30 tokens/sec** (Q5_K_M) to a crawling **4.08 tokens/sec** (Q8_0). 
*   **Takeaway**: Running a highly quantized model that fits completely in VRAM is always faster than running a higher-precision model that spills into system RAM.

### Hardware Memory Bandwidth Ceilings
LLM generation is memory-bandwidth bound. The speed of loading model weights from memory to the processing cores sets the performance ceiling.
*   Our sweeps show that the decode throughput of `Q4_K_M` (33.63 t/s) is **14.8% faster** than `Q5_K_M` (29.30 t/s).
*   This difference is directly proportional to weight size: `Q4_K_M` weights are ~12% smaller than `Q5_K_M` weights. 
*   Since the GPU's memory bus bandwidth is fixed (192 GB/s), loading the smaller 4-bit weights takes less time, directly translating to a proportional increase in generation speed.

---

## 4. Engineering Recommendations
1.  **Use Q4_K_M as the Default**: It provides the best balance of interactive speed (~33.6 tokens/sec) and VRAM safety buffer, making it the most stable configuration for a 6 GB VRAM budget.
2.  **Use Q5_K_M only for short context sessions**: If your task requires higher reasoning accuracy (e.g., code generation or complex math) and your context window is restricted (under 1024 tokens), `Q5_K_M` provides improved perplexity while staying within VRAM limits.
3.  **Avoid Q8_0 on 6 GB VRAM**: The weight size guarantees PCIe swapping, rendering inference too slow for interactive use.
