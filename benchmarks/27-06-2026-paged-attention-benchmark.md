# Benchmark Report: PagedAttention and Prompt Caching Efficiency under VRAM Constraints

This report compiles the benchmarking results comparing prompt evaluation latency and VRAM utilization with and without Prompt Caching (Shared Pages) across different quantization profiles (`Q4_K_M`, `Q5_K_M`, and `Q8_0`) of the `Qwen2.5-7B-Instruct` model.

---

## 1. Executive Summary

Prompt caching (analogous to PagedAttention page sharing) optimizes prefill latency by caching the KV states of static prompt prefixes.
*   **Speedup**: For fully offloaded quants (`Q4_K_M` and `Q5_K_M`), enabling prompt caching yields a **35x to 37x speedup** on warm prompt hits, slashing prefill latency from **~1.9s** down to **~0.05s** for a 2000-token prefix.
*   **VRAM Spillage Barrier**: When evaluating `Q8_0` with full offload (`ngl=33`), the model footprint hits the physical 6GB VRAM limit (allocating ~6.11 GB). This triggers system-level memory swapping, increasing prefill latency to **18.3s** (a 9.5x slowdown compared to Q4/Q5).
*   **CPU Offloading Bottleneck**: Offloading fewer layers (`ngl=15`) avoids VRAM spillage, but CPU-bound attention execution reduces warm cache speedup to **~10x** (0.56s).

---

## 2. Experimental Setup

*   **Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM), Intel Core i7, 16GB RAM.
*   **Baseline Model**: `Qwen2.5-7B-Instruct` in GGUF formats.
*   **Prompt Configuration**:
    *   **Prefix**: 2000 tokens (static context).
    *   **Suffix**: 10 tokens (variable user query).
    *   **Total Prompt Size**: 2010 tokens.
*   **Software**: `llama-cpp-python` with `LlamaRAMCache` (512 MB capacity).

---

## 3. Quantitative Results Table

| Quantization | GPU Layers (`ngl`) | Cache Enabled | Cold Prefill (s) | Warm Prefill (s) | Speedup Factor | Peak VRAM (MB) | VRAM Delta (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q4_K_M** | 33 | False | 1.9832 | 1.6611 | 1.19x | 5000.53 | 80.00 |
| **Q4_K_M** | 33 | True | 1.8597 | 0.0500 | **37.22x** | 5000.53 | 80.00 |
| **Q5_K_M** | 33 | False | 1.9675 | 1.7043 | 1.15x | 5660.53 | 80.00 |
| **Q5_K_M** | 33 | True | 1.8863 | 0.0537 | **35.16x** | 5660.53 | 80.00 |
| **Q8_0** | 33 | False | 18.3228 | 17.8858 | 1.02x | 6130.53 | 20.00 |
| **Q8_0** | 33 | True | 18.1775 | 0.6177 | **29.43x** | 6130.53 | 20.00 |
| **Q8_0** | 15 | False | 8.5075 | 3.3623 | 2.53x | 4592.53 | 94.00 |
| **Q8_0** | 15 | True | 5.5709 | 0.5583 | **9.98x** | 4592.53 | 94.00 |

---

## 4. Key Findings and System Analysis

### A. The Mechanics of Caching Speedup
When prompt caching is active, the model bypasses prompt evaluation for the prefix and loads the stored keys and values directly into its KV buffer. Rather than executing $2010$ attention projections, it only projects the $10$ new suffix tokens. This drops prefill latency from **~1.9s** to **~0.05s**.

### B. VRAM Spillage Threshold (The 6GB Memory Wall)
For `Q8_0` with `ngl=33`, the combined weight, context, and driver allocations exceed the GPU's physical VRAM limit. 
> [!WARNING]
> When allocations exceed 6144 MB, Windows initiates virtualized memory paging (swapping GPU allocations to system RAM over the PCIe bus). This drops GPU compute utilization to near-zero as execution stalls waiting for memory transfers, raising prefill latency from **1.9s** to **18.3s** (a **960% slowdown**).

### C. CPU Offloading vs. Memory-Bandwidth Bottleneck
By restricting `ngl=15` for `Q8_0`, the active VRAM footprint is kept at **4592.53 MB** (safely below the 6GB threshold). However:
*   The layers left on the CPU must execute prompt evaluation sequentially without CUDA acceleration.
*   This drops the cold prefill speed from 18.3s (spilled GPU) to **5.57s** (CPU offloaded), which is a net win for cold starts.
*   However, during warm caching runs, the CPU must still perform memory transfers and process local attention maps, limiting the caching speedup to **9.98x** (0.5583s) compared to the **37x** speedup achieved on the GPU.

---

## Related Documentation
*   [Theory Reference: KV Cache Sizing and Memory Layouts](../docs/kv-cache.md)
*   [Learning Report: Hardware Setup and Compilation](../learnings/learning-hardware-setup.md)
