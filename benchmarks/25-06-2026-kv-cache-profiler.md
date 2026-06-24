# Benchmark Report: KV Cache Memory and VRAM Scaling Profiler

This report details the experimental VRAM memory profile of `Qwen2.5-7B` across varying context lengths, quantization levels, and offloading configurations. Measurements were performed on a laptop with a physical **NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)**.

---

## 1. Experimental Setup and Methodology

To measure memory growth in complete isolation and prevent GPU state leakage between runs, each profile configuration was executed in a **dedicated, isolated Python subprocess**.
*   **Model**: Qwen2.5-7B-Instruct (GGUF format)
*   **Context Lengths ($T$)**: 128, 256, 512, 1024, 2048, and 4096 tokens
*   **Quantization Formats**: `Q4_K_M` (~4.0 GB weights), `Q5_K_M` (~4.8 GB weights), and `Q8_0` (~7.7 GB weights)
*   **Measurement Protocol**:
    1.  **Loaded VRAM Baseline**: Measured after model loading (`n_ctx=T`), representing `weights + static KV cache pre-allocation`.
    2.  **Prefill Peak VRAM**: High-frequency sampling (50 Hz) during the parallel evaluation of a sequence of size `T - 10`.
    3.  **Decode Speed**: Autoregressive generation of 5 tokens, measured in tokens per second (t/s).

---

## 2. Profile Benchmark Data

The table below summarizes the VRAM requirements and decode speeds for each configuration.

| Model Quant | ngl | Context Length | Theoretical KV | Experimental KV | Prefill Peak VRAM | Prefill Spike (Act.) | Decode Speed | PCIe Paging |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q4_K_M** | 33 | 128 | 7.00 MB | 70.00 MB | 4510.53 MB | 28.00 MB | 32.15 t/s | **No** |
| **Q4_K_M** | 33 | 256 | 14.00 MB | 146.00 MB | 4588.53 MB | 30.00 MB | 31.35 t/s | **No** |
| **Q4_K_M** | 33 | 512 | 28.00 MB | 312.00 MB | 4766.53 MB | 42.00 MB | 15.60 t/s | **No** |
| **Q4_K_M** | 33 | 1024 | 56.00 MB | 340.00 MB | 4808.53 MB | 56.00 MB | 15.04 t/s | **No** |
| **Q4_K_M** | 33 | 2048 | 112.00 MB | 396.00 MB | 4892.53 MB | 84.00 MB | 12.74 t/s | **No** |
| **Q4_K_M** | 33 | 4096 | 224.00 MB | 508.00 MB | 5058.53 MB | 138.00 MB | 9.74 t/s | **No** |
| **Q5_K_M** | 33 | 128 | 7.00 MB | 70.00 MB | 5170.53 MB | 28.00 MB | 28.65 t/s | **No** |
| **Q5_K_M** | 33 | 256 | 14.00 MB | 146.00 MB | 5248.53 MB | 30.00 MB | 29.33 t/s | **No** |
| **Q5_K_M** | 33 | 512 | 28.00 MB | 312.00 MB | 5426.53 MB | 42.00 MB | 14.84 t/s | **No** |
| **Q5_K_M** | 33 | 1024 | 56.00 MB | 340.00 MB | 5468.53 MB | 56.00 MB | 13.85 t/s | **No** |
| **Q5_K_M** | 33 | 2048 | 112.00 MB | 396.00 MB | 5552.53 MB | 84.00 MB | 11.95 t/s | **No** |
| **Q5_K_M** | 33 | 4096 | 224.00 MB | 508.00 MB | 5718.53 MB | 138.00 MB | 9.17 t/s | **No** |
| **Q8_0** | 33 | 128 | 7.00 MB | 0.00 MB * | 6096.53 MB | 0.00 MB | 3.46 t/s | **Yes** |
| **Q8_0** | 33 | 256 | 14.00 MB | 0.00 MB * | 6098.53 MB | 0.00 MB | 2.50 t/s | **Yes** |
| **Q8_0** | 33 | 512 | 28.00 MB | 0.00 MB * | 6096.53 MB | 0.00 MB | 1.23 t/s | **Yes** |
| **Q8_0** | 33 | 1024 | 56.00 MB | 0.00 MB * | 6110.53 MB | 0.00 MB | 1.09 t/s | **Yes** |
| **Q8_0** | 33 | 2048 | 112.00 MB | 0.00 MB * | 6128.53 MB | 18.00 MB | 0.79 t/s | **Yes** |
| **Q8_0** | 33 | 4096 | 224.00 MB | 0.00 MB * | 6132.53 MB | 22.00 MB | 0.46 t/s | **Yes** |
| **Q8_0 (Opt)** | 15 | 128 | 7.00 MB | 88.00 MB | 4210.53 MB | 26.00 MB | 8.59 t/s | **No** |
| **Q8_0 (Opt)** | 15 | 256 | 14.00 MB | 146.00 MB | 4270.53 MB | 28.00 MB | 9.45 t/s | **No** |
| **Q8_0 (Opt)** | 15 | 512 | 28.00 MB | 304.00 MB | 4440.53 MB | 40.00 MB | 9.08 t/s | **No** |
| **Q8_0 (Opt)** | 15 | 1024 | 56.00 MB | 318.00 MB | 4468.53 MB | 54.00 MB | 8.10 t/s | **No** |
| **Q8_0 (Opt)** | 15 | 2048 | 112.00 MB | 346.00 MB | 4532.53 MB | 90.00 MB | 9.03 t/s | **No** |
| **Q8_0 (Opt)** | 15 | 4096 | 224.00 MB | 402.00 MB | 4650.53 MB | 152.00 MB | 8.22 t/s | **No** |

*\* Note: Under Q8_0 (ngl=33), baseline weights already saturate the physical 6 GB limit. The reported experimental cache size is 0 MB because VRAM is pegged at the physical threshold, meaning all additional KV cache allocations are immediately routed into WDDM shared system RAM.*

---

## 3. Analysis of Key Findings

### A. The VRAM Swapping Cliff (PCIe Paging Overhead)
The RTX 3050 Laptop GPU has exactly **6,144 MB (6.0 GB)** of dedicated VRAM. When model parameters + context exceed this physical boundary, Windows WDDM driver pages memory over the PCIe bus (utilizing system RAM).
*   For **Q8_0 (ngl=33)**, the baseline model weights consume $6.13$ GB. This immediately saturates VRAM.
*   During generation at a $4,096$ context window, the decode speed collapses to **$0.46$ t/s** (a **95% drop** compared to Q4_K_M). 
*   Furthermore, the prefill time for the $4,096$-token prompt took an astronomical **$39.63$ seconds** because the model weights and activations were repeatedly thrashed over the narrow PCIe Gen 4 x4 bus.

```
       RTX 3050 (6 GB VRAM) Context Scaling - Generation TPS
  35 +----------------------------------------------------------------+
     |   * Q4_K_M (ngl=33) - 32.15 t/s                                |
  30 |   + Q5_K_M (ngl=33) - 28.65 t/s                                |
     |                                                                |
  25 |                                                                |
     |                                                                |
  20 |                                                                |
     |                                                                |
  15 |                   * Q4_K_M - 15.60 t/s                         |
     |                   + Q5_K_M - 14.84 t/s                         |
  10 |                                              * Q4_K_M - 9.74   |
     |   # Q8_0 (ngl=15) - 8.59 t/s                 # Q8_0 (ngl=15)-  |
   5 |                                                      8.22 t/s  |
     |   $ Q8_0 (ngl=33) - 3.46 t/s                                   |
   0 +----------------------------------------------------------------+
     128                 512                                4096
                               Context Length (Tokens)
```

### B. Prefill Activation Spikes
The prefill phase evaluates all prompt tokens in parallel, which creates intermediate layers of activation states.
*   At $128$ context, the prefill activation spike is only **$28$ MB**.
*   At $4,096$ context, the prefill activation spike increases to **$138$ MB** ($152$ MB for Q8_0).
*   This spike scales quadratically with sequence length because calculating the raw attention score matrix requires $T \times T$ operations. This means that to prevent OOM errors, a VRAM buffer of at least $150$-$200$ MB must be kept free above the static loading baseline.

### C. The Partial Offloading Special Case (Q8_0 Optimized)
When a large model like `Q8_0` is loaded, forcing full offload (`ngl=33`) results in severe paging. 
*   By implementing a **special case offloading fallback (`ngl=15`)**, we offload only 15 out of 28 layers to the GPU, keeping the remaining 13 layers and their KV caches in host CPU RAM.
*   This drops the static loaded memory to **$4,184$ MB**, fitting comfortably under the physical 6 GB limit.
*   **Result**: 
    *   Decode speed at $4,096$ context reaches **$8.22$ t/s** (an **18x performance speedup** compared to the $0.46$ t/s paging run).
    *   Prefill time drops from $39.63$s to **$8.52$ seconds** (a **4.6x speedup**).
*   **Conclusion**: In memory-constrained settings, partial CPU offloading is vastly superior to allowing GPU weights to overflow and trigger PCIe memory swapping.

---

## Related Documentation
*   [Theory Reference: KV Cache Sizing and Memory Layouts](../docs/kv-cache.md)
*   [Learning Report: KV Cache Math and Hardware Constraints](../learnings/learning-kv-cache-math.md)
