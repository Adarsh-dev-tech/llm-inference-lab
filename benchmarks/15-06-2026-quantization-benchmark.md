# [RTX 3050 Laptop] Qwen2.5-7B Quantization Performance & Throughput Sweeps

Date: 15-06-2026

## Objective
Evaluate the impact of model weight quantization bit-widths (Q4_K_M, Q5_K_M, Q8_0) on prompt processing speed (prefill TPS), token generation speed (decode TPS), latency (TTFT, ITL), and system resource usage (host RAM, GPU VRAM) under a resource-constrained 6 GB VRAM budget.

## Research Question
Does increasing the quantization bit-width from 4-bit (Q4_K_M) to 5-bit (Q5_K_M) and 8-bit (Q8_0) cause a performance cliff (drop in decode throughput) due to VRAM overflow and host memory paging on a 6 GB laptop GPU?

---

## Hardware Configuration
*   **CPU**: Intel Core i5-12450HX (8 cores / 12 threads)
*   **RAM**: 16 GB DDR5 Single-Channel (4800 MHz, 38.4 GB/s theoretical bandwidth)
*   **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB GDDR6, 128-bit bus, 192 GB/s bandwidth, PCIe Gen4 x8 interface)
*   **Cooling**: Standard laptop active dual-fan cooling

## Software Configuration
*   **OS**: Windows 11 Home
*   **Inference Engine**: `llama-cpp-python v0.3.28` compiled with CUDA support (`LLAMA_CUDA=ON`)
*   **CUDA version**: CUDA 13.0 / Driver 581.86

## Model Information
*   **Model**: `Qwen2.5-7B-Instruct-GGUF`
*   **Parameters**: 7.61 Billion
*   **Layers**: 32 transformer layers (+ 1 input/output layer, total 33 layer offloads)
*   **Quantizations Tested**:
    *   `Q4_K_M` (mixed 4-bit, ~4.80 GB weight footprint)
    *   `Q5_K_M` (mixed 5-bit, ~5.43 GB weight footprint)
    *   `Q8_0` (standard 8-bit, ~8.00 GB weight footprint)

## Benchmark Configuration
*   **Prompts Tested**:
    *   `short.txt` (~10 tokens input)
    *   `medium.txt` (~25 tokens input)
    *   `long.txt` (~65 tokens input)
*   **GPU Offload Count (`ngl`)**: 33 layers (full offload attempt)
*   **Generation Parameters**: Temperature = 0.7, Repetition Penalty = 1.1, Max Tokens = 2048 (allowed to naturally run to EOS)
*   **Iterations**: 3 runs per configuration (discarding warm-up calculations for metrics)

---

## Raw Results & Consolidation

Here is the aggregate performance across the configurations (averages over non-warmup runs):

### 1. Q4_K_M Quantization
*   **Model Weight VRAM Footprint**: 4.78 GB (Total loaded VRAM: 4.92 GB)
*   **System RAM usage increase**: ~4.12 GB

| Metric | Short Prompt (10 tok) | Medium Prompt (25 tok) | Long Prompt (65 tok) |
| :--- | :--- | :--- | :--- |
| **Generated Tokens** | 12 tokens | 100 tokens | 943 tokens |
| **Decode Throughput (TPS)** | 33.90 t/s | 33.75 t/s | 33.23 t/s |
| **Prefill Throughput (TPS)** | 318.89 t/s | 760.00 t/s | 1215.16 t/s |
| **Time-to-First-Token (TTFT)**| 31.85 ms | 32.90 ms | 38.37 ms |
| **Inter-Token Latency (ITL)** | 29.48 ms | 29.62 ms | 30.10 ms |
| **Avg GPU utilization** | ~94.0% | ~93.1% | ~91.3% |
| **Avg GPU Mem Controller** | ~99.0% | ~97.6% | ~98.8% |

---

### 2. Q5_K_M Quantization
*   **Model Weight VRAM Footprint**: 5.44 GB (Total loaded VRAM: 5.58 GB)
*   **System RAM usage increase**: ~5.21 GB

| Metric | Short Prompt (10 tok) | Medium Prompt (25 tok) | Long Prompt (65 tok) |
| :--- | :--- | :--- | :--- |
| **Generated Tokens** | 53 tokens | 285 tokens | 982 tokens |
| **Decode Throughput (TPS)** | 29.54 t/s | 29.30 t/s | 29.06 t/s |
| **Prefill Throughput (TPS)** | 280.09 t/s | 634.68 t/s | 1025.13 t/s |
| **Time-to-First-Token (TTFT)**| 35.70 ms | 39.39 ms | 41.55 ms |
| **Inter-Token Latency (ITL)** | 33.86 ms | 34.12 ms | 34.41 ms |
| **Avg GPU utilization** | ~95.9% | ~94.2% | ~91.8% |
| **Avg GPU Mem Controller** | ~100.0% | ~97.9% | ~99.8% |

---

### 3. Q8_0 Quantization (VRAM Over-allocation)
*   **Model Weight VRAM Footprint**: 5.97 GB (Total loaded VRAM: 6.11 GB - saturated)
*   **System RAM usage increase**: ~9.06 GB (spills into host RAM)

| Metric | Short Prompt (10 tok) | Medium Prompt (25 tok) | Long Prompt (65 tok) |
| :--- | :--- | :--- | :--- |
| **Generated Tokens** | 38 tokens | 279 tokens | 1075 tokens |
| **Decode Throughput (TPS)** | 4.22 t/s | 4.23 t/s | 3.81 t/s |
| **Prefill Throughput (TPS)** | 41.97 t/s | 103.25 t/s | 81.41 t/s |
| **Time-to-First-Token (TTFT)**| 238.25 ms | 242.08 ms | 431.85 ms |
| **Inter-Token Latency (ITL)** | 236.85 ms | 235.89 ms | 262.28 ms |
| **Avg GPU utilization** | ~20.3% | ~25.2% | ~29.7% |
| **Avg GPU Mem Controller** | ~17.5% | ~19.1% | ~18.3% |

---

## Statistical Summary & Comparison

| Quantization Format | Mean Decode Throughput | Prefill TPS (Long Prompt) | Mean ITL (ms) | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- |
| **Q4_K_M** | **33.63 tokens/sec** | **1215.16 t/s** | **29.73 ms** | 4.92 GB |
| **Q5_K_M** | **29.30 tokens/sec** | **1025.13 t/s** | **34.13 ms** | 5.58 GB |
| **Q8_0** | **4.08 tokens/sec** | **81.41 t/s** | **245.01 ms** | 6.11 GB |

---

## Analysis

### 1. The VRAM Cliff & Performance Collapse
The experiment shows a dramatic **8.2x performance drop** in decode speed (from 33.63 t/s to 4.08 t/s) and a **15x drop** in prefill speed when moving from `Q5_K_M` to `Q8_0`. 
*   Because `Q8_0` weights (~8.00 GB) exceed the physical 6 GB limit of the RTX 3050 Laptop GPU, the Windows CUDA driver (WDDM) is forced to dynamically page memory blocks.
*   The GPU cannot fit the entire model graph in local GDDR6 memory. Consequently, weights are continuously swapped back and forth between VRAM and system RAM over the slow PCIe bus.
*   This paging mechanism stalls the GPU's Streaming Multiprocessors (SMs), as evidenced by the drop in average GPU Memory Controller utilization (from 99.8% to 18.3%) and lower power draw (from 74W to 33W). The GPU is mostly idle, waiting for memory transfers over the PCIe interface.

### 2. Prefill vs. Decode Scaling
*   **Prefill Phase (Compute-Bound)**: Processed in parallel. Throughput scales significantly with prompt length (e.g., Q4_K_M goes from 318 t/s on 10 tokens to 1215 t/s on 65 tokens) as matrix-multiplication operations occupy more Tensor Cores.
*   **Decode Phase (Memory-Bound)**: Processed autoregressively (one token at a time). Speed remains highly stable regardless of prompt length, because weights must be loaded sequentially for each generated token.

---

## Unexpected Findings

During initial runs, background instances of **LM Studio** occupied ~1.5 GB of GPU VRAM. 
*   This caused the `Q8_0` model to drop to an even lower generation throughput (~0.1 to 0.2 tokens/sec) and loop endlessly due to severe memory thrashing. 
*   Once all background LM Studio processes were terminated, the driver paging stabilized, allowing the `Q8_0` model to run at a consistent but still severely bottlenecked ~3.8 to 4.2 tokens/sec. 
*   This highlights the critical importance of Rule 1 of our methodology: **close all background applications** before conducting local hardware benchmarks.

---

## Conclusions
1.  **Hypothesis Confirmed**: Exceeding the VRAM boundary triggers a performance collapse. `Q8_0` is completely unusable for interactive applications on a 6 GB VRAM budget due to PCIe paging overhead.
2.  **Q5_K_M vs. Q4_K_M**: `Q5_K_M` fits inside the 6 GB VRAM limit (utilizing ~5.58 GB total) and runs at 29.30 tokens/sec. It provides a viable alternative to `Q4_K_M` (33.63 tokens/sec) if higher accuracy/perplexity is required.
3.  **Optimal Default**: `Q4_K_M` remains the recommended default, leaving ~1.08 GB VRAM headroom for the KV cache to expand under longer context sequences without hitting the VRAM cliff.

## Future Experiments
*   Evaluate KV cache memory allocation sizing with `llama.cpp` context sweeps (1024 to 4096 context tokens) to determine the exact point where `Q5_K_M` triggers VRAM overflow.
*   Test split-offloading configurations (`-ngl <33`) on the `Q8_0` model to see if pinning a fixed set of layers to VRAM yields better throughput than letting WDDM perform dynamic paging.

## Related Learnings
*   [Learning Report: Quantization and Memory Bandwidth Limits](../learnings/learning-quantization-differences.md)

## Related Theory Documents
*   [Theory Reference: Quantization Precision Formats](../docs/quantization.md)
*   [Theory Reference: Hardware Benchmarking Methodology](../docs/benchmarking-methodology.md)
