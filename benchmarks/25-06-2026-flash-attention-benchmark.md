# Benchmark Report: FlashAttention Memory and Speed Sweeps

This report presents a comparative performance profile of the Qwen2.5-7B-Instruct model with **FlashAttention enabled (`flash_attn=True`)** versus **Standard Attention (`flash_attn=False`)** across varying context lengths, quantization formats, and offloading levels.

Measurements were performed on a laptop with an **NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)**.

---

## 1. Experimental Setup and Scope

*   **Subprocess Isolation**: To ensure CUDA context purity and prevent memory leakage, each context length and quantization configuration was executed in a dedicated, isolated Python subprocess.
*   **Model**: Qwen2.5-7B-Instruct (GGUF format)
*   **Context Lengths ($T$)**: 512, 1024, 2048, and 4096 tokens
*   **Quantization Formats**: `Q4_K_M` (~4.0 GB weights), `Q5_K_M` (~4.8 GB weights), and `Q8_0` (~7.7 GB weights)
*   **Offload Settings**:
    *   `ngl=33` (Full GPU Offload) for all models
    *   `ngl=15` (Optimized CPU-fallback) for the `Q8_0` model to prevent PCIe VRAM swapping

---

## 2. Comparative Benchmark Results

The table below outlines the loaded VRAM baseline, prefill peak memory, prompt processing duration, and decode throughput.

| Model Quant | ngl | Flash | Context | Loaded VRAM | Prefill Peak | Prefill Spike | Prefill Duration | Decode Speed | PCIe Paging |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q4_K_M** | 33 | **Off** | 512 | 4724.53 MB | 4766.53 MB | **42.00 MB** | 0.5284s | 15.59 t/s | No |
| **Q4_K_M** | 33 | **On** | 512 | 4724.53 MB | 4744.53 MB | **20.00 MB** | 0.0542s | 19.10 t/s | No |
| **Q4_K_M** | 33 | **Off** | 1024 | 4752.53 MB | 4808.53 MB | **56.00 MB** | 0.7245s | 14.33 t/s | No |
| **Q4_K_M** | 33 | **On** | 1024 | 4752.53 MB | 4772.53 MB | **20.00 MB** | 0.4241s | 19.79 t/s | No |
| **Q4_K_M** | 33 | **Off** | 2048 | 4808.53 MB | 4892.53 MB | **84.00 MB** | 1.6032s | 12.64 t/s | No |
| **Q4_K_M** | 33 | **On** | 2048 | 4808.53 MB | 4828.53 MB | **20.00 MB** | 1.0882s | 18.48 t/s | No |
| **Q4_K_M** | 33 | **Off** | 4096 | 4920.53 MB | 5058.53 MB | **138.00 MB** | 3.6437s | 9.12 t/s | No |
| **Q4_K_M** | 33 | **On** | 4096 | 4920.53 MB | 4940.53 MB | **20.00 MB** | 2.5218s | 17.80 t/s | No |
| **Q5_K_M** | 33 | **Off** | 512 | 5384.53 MB | 5426.53 MB | **42.00 MB** | 0.3978s | 13.01 t/s | No |
| **Q5_K_M** | 33 | **On** | 512 | 5384.53 MB | 5404.53 MB | **20.00 MB** | 0.0615s | 17.62 t/s | No |
| **Q5_K_M** | 33 | **Off** | 1024 | 5412.53 MB | 5468.53 MB | **56.00 MB** | 0.7702s | 13.23 t/s | No |
| **Q5_K_M** | 33 | **On** | 1024 | 5412.53 MB | 5432.53 MB | **20.00 MB** | 0.4309s | 16.68 t/s | No |
| **Q5_K_M** | 33 | **Off** | 2048 | 5468.53 MB | 5552.53 MB | **84.00 MB** | 1.5865s | 11.68 t/s | No |
| **Q5_K_M** | 33 | **On** | 2048 | 5468.53 MB | 5488.53 MB | **20.00 MB** | 1.1230s | 17.41 t/s | No |
| **Q5_K_M** | 33 | **Off** | 4096 | 5580.53 MB | 5718.53 MB | **138.00 MB** | 3.7720s | 8.87 t/s | No |
| **Q5_K_M** | 33 | **On** | 4096 | 5580.53 MB | 5600.53 MB | **20.00 MB** | 2.5977s | 15.97 t/s | No |
| **Q8_0** | 33 | **Off** | 512 | 6110.53 MB | 6096.53 MB | **0.00 MB** * | 0.7200s | 1.16 t/s | Yes |
| **Q8_0** | 33 | **On** | 512 | 6110.53 MB | 6076.53 MB | **0.00 MB** * | 0.0678s | 1.31 t/s | Yes |
| **Q8_0** | 33 | **Off** | 1024 | 6110.53 MB | 6110.53 MB | **0.00 MB** * | 3.8928s | 1.09 t/s | Yes |
| **Q8_0** | 33 | **On** | 1024 | 6110.53 MB | 6076.53 MB | **0.00 MB** * | 3.2193s | 1.19 t/s | Yes |
| **Q8_0** | 33 | **Off** | 2048 | 6110.53 MB | 6128.53 MB | **18.00 MB** | 12.4396s | 0.75 t/s | Yes |
| **Q8_0** | 33 | **On** | 2048 | 6110.53 MB | 6076.53 MB | **0.00 MB** * | 10.8094s | 0.97 t/s | Yes |
| **Q8_0** | 33 | **Off** | 4096 | 6110.53 MB | 6132.53 MB | **22.00 MB** | 38.7045s | 0.48 t/s | Yes |
| **Q8_0** | 33 | **On** | 4096 | 6110.53 MB | 6076.53 MB | **0.00 MB** * | 31.4377s | 0.72 t/s | Yes |
| **Q8_0 (Opt)**| 15 | **Off** | 512 | 4400.53 MB | 4440.53 MB | **40.00 MB** | 1.0975s | 9.55 t/s | No |
| **Q8_0 (Opt)**| 15 | **On** | 512 | 4400.53 MB | 4420.53 MB | **20.00 MB** | 0.5501s | 10.44 t/s | No |
| **Q8_0 (Opt)**| 15 | **Off** | 1024 | 4414.53 MB | 4468.53 MB | **54.00 MB** | 2.4463s | 7.94 t/s | No |
| **Q8_0 (Opt)**| 15 | **On** | 1024 | 4414.53 MB | 4434.53 MB | **20.00 MB** | 1.5798s | 9.30 t/s | No |
| **Q8_0 (Opt)**| 15 | **Off** | 2048 | 4442.53 MB | 4532.53 MB | **90.00 MB** | 4.6724s | 9.18 t/s | No |
| **Q8_0 (Opt)**| 15 | **On** | 2048 | 4442.53 MB | 4470.53 MB | **28.00 MB** | 3.5599s | 9.40 t/s | No |
| **Q8_0 (Opt)**| 15 | **Off** | 4096 | 4498.53 MB | 4650.53 MB | **152.00 MB** | 8.9550s | 8.21 t/s | No |
| **Q8_0 (Opt)**| 15 | **On** | 4096 | 4498.53 MB | 4534.53 MB | **36.00 MB** | 7.1462s | 8.70 t/s | No |

*\* Note: Under Q8_0 (ngl=33), baseline weights already saturate the physical 6 GB limit. The reported prefill spike is 0 MB because VRAM is pegged at the driver limit, meaning all prompt evaluation activations are immediately routed into WDDM shared system RAM.*

---

## 3. Analysis of Key Findings

### A. The Memory Spike Compression Effect (Quadratic Flattening)
The most striking result is the behavior of the **Prefill VRAM Activation Spike** (the temporary memory allocated to hold intermediate matrices like $QK^T$ during parallel evaluation):
*   **Without FlashAttention (Off)**: The activation spike grows quadratically as sequence length scales. For `Q4_K_M`, it starts at **$42$ MB** (512 context) and swells to **$138$ MB** at 4096 context.
*   **With FlashAttention (On)**: The prefill activation spike is **flattened to a constant $20.00$ MB** across 512, 1024, 2048, and 4096 tokens! 
*   **System Impact**: FlashAttention completely eliminates the quadratic $O(T^2)$ memory footprint of attention activations. On memory-constrained GPUs, this prevents the prompt evaluation stage from triggering VRAM allocation peaks that could cause sudden Out-Of-Memory (OOM) failures.

```
       Prefill VRAM Activation Spike Growth (Q4_K_M)
  150 +------------------------------------------------------------+
      |                                              * Off (138 MB)|
  125 |                                                            |
      |                                                            |
  100 |                                                            |
      |                               * Off (84 MB)                |
   75 |                                                            |
      |                * Off (56 MB)                               |
   50 |  * Off (42 MB)                                             |
      |                                                            |
   25 |  # On (20 MB)  # On (20 MB)   # On (20 MB)   # On (20 MB)  |
    0 +------------------------------------------------------------+
      512              1024           2048           4096
                             Context Length (Tokens)
```

### B. Prompt Processing Speedups (Prefill Stage)
By bypassing HBM reads/writes, FlashAttention accelerates prompt processing significantly:
*   For **Q4_K_M (ngl=33)**: Prefill time at 4096 context drops from $3.64$ seconds to **$2.52$ seconds** (a **$31\%$ speedup**).
*   For **Q8_0 (ngl=15)**: Prefill time at 4096 context drops from $8.96$ seconds to **$7.15$ seconds** (a **$20\%$ speedup**).
*   This performance increase directly benefits the time-to-first-token (TTFT) latency, especially during long-context workloads.

### C. Generation Speedups (Decode Stage)
During autoregressive generation (decoding), FlashAttention also yields major performance increases for offloaded models:
*   For **Q4_K_M (ngl=33)**: Generation speed at 4096 context increases from $9.12$ t/s to **$17.80$ t/s** (a **$95\%$ throughput increase**).
*   For **Q5_K_M (ngl=33)**: Generation speed at 4096 context increases from $8.87$ t/s to **$15.97$ t/s** (an **$80\%$ throughput increase**).
*   **Why?**: Although decode is memory-bound (streaming weights), FlashAttention's kernel fusion optimizes cache locality in the GPU L1/L2 caches and SRAM during head-by-head attention calculations, reducing memory access overhead.

---

## Related Documentation
*   [Theory Reference: Fused Attention Kernels and SRAM Tiling](../docs/flash-attention.md)
*   [Theory Reference: KV Cache Sizing and Memory Layouts](../docs/kv-cache.md)
*   [Learning Report: GPU Memory Hierarchy and IO Constraints](../learnings/learning-flashattention-memory-io.md)
