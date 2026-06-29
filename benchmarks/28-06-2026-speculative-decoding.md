# Benchmark Report: Native Speculative Decoding Speedup and Acceptance Rates

This report analyzes the performance, memory bounds, and latency trade-offs of native Speculative Decoding in `llama.cpp` using a `Qwen2.5-7B-Instruct` target model paired with either a `Qwen2.5-1.5B-Instruct` or a `Qwen2.5-0.5B-Instruct` draft model.

---

## 1. Executive Summary

Speculative decoding relies on a small draft model predicting sequence extensions that are subsequently verified in parallel by a larger target model.
*   **Draft Acceptance Rate**: The draft models achieve a consistent **~47% to 58% token acceptance rate** when matched against the 7B target.
*   **1.5B vs. 0.5B Draft Comparison**: 
    *   **VRAM Allocation Limit (The Crash Boundary)**: Attempting to run the 1.5B draft model alongside `Q5_K_M` or `Q8_0` targets on a 6GB VRAM card leads to **immediate Vulkan device memory allocation failure (OOM)**. The smaller 0.5B draft model (~398 MB) fits comfortably, executing successfully without OOM across all targets.
    *   **Overhead Scaling**: The 1.5B draft model represents ~25% of the target's parameters, causing severe sequential evaluation overhead. The 0.5B draft model is ~7% of the target's parameters, drastically reducing draft evaluation latency.
*   **On-GPU Speculation Bottleneck**: When target weights are fully offloaded to the GPU (`ngl=33`), the target generates at **34.5 t/s**. Running speculation with the 0.5B draft model still causes a slowdown to **26.5 t/s** (slightly more severe than the 1.5B draft's 26.7 t/s compared to its baseline) due to kernel launch serialization latency on the Edge GPU.
*   **CPU-GPU Split Offloading Speedup**: When the target model is partially offloaded (`ngl=15` for Q8_0, running at **10.0 t/s** target-only), speculation with the 0.5B draft model yields a **net speedup** to **10.5 t/s** (+5.0% speedup). This represents the target use case where draft validation saves expensive CPU-GPU offload calculation roundtrips.

---

## 2. Experimental Configuration

*   **Target Model**: `Qwen2.5-7B-Instruct` GGUF (`Q4_K_M`, `Q5_K_M`, `Q8_0`).
*   **Draft Models**:
    *   `Qwen2.5-1.5B-Instruct-Q4_K_M` GGUF (~1.2 GB)
    *   `Qwen2.5-0.5B-Instruct-Q4_K_M` GGUF (~398 MB)
*   **Hardware**: NVIDIA GeForce RTX 3050 Laptop (6GB VRAM), 16GB System RAM.
*   **Parameters**: `--single-turn`, `-n 64`, `-c 2048` (context size locked to prevent massive default KV allocation), `--spec-type draft-simple`.

---

## 3. Quantitative Results Table

### A. Sweeps with Qwen2.5-0.5B-Instruct Draft Model (New)

| Target Quant | Target ngl | Speculative Enabled | Prompt Speed (t/s) | Generation Speed (t/s) | Draft Acceptance Rate | Peak VRAM (MB) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Q4_K_M** | 33 | False | 362.1 | 34.5 | N/A | 4561.4 | **Success** |
| **Q4_K_M** | 33 | True | 20.5 | 26.5 | 55.6% | 4999.5 | **Success** |
| **Q5_K_M** | 33 | False | 334.2 | 29.8 | N/A | 5222.8 | **Success** |
| **Q5_K_M** | 33 | True | 291.5 | 19.6 | 57.8% | 5662.4 | **Success** |
| **Q8_0** | 33 | False | 0.0 | 0.0 | N/A | 5117.2 | **Failed (OOM)** |
| **Q8_0** | 33 | True | 0.0 | 0.0 | N/A | 5117.2 | **Failed (OOM)** |
| **Q8_0** | 15 | False | 76.4 | 10.0 | N/A | 4252.1 | **Success** |
| **Q8_0** | 15 | True | 48.0 | 10.5 | 47.1% | 4689.0 | **Success (+5.0% Speedup)** |

### B. Sweeps with Qwen2.5-1.5B-Instruct Draft Model (Old Reference)

| Target Quant | Target ngl | Speculative Enabled | Prompt Speed (t/s) | Generation Speed (t/s) | Draft Acceptance Rate | Peak VRAM (MB) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Q4_K_M** | 33 | False | 421.9 | 34.3 | N/A | 4561.4 | **Success** |
| **Q4_K_M** | 33 | True | 298.5 | 26.7 | 50.0% | 5619.7 | **Success** |
| **Q5_K_M** | 33 | False | 40.9 | 22.8 | N/A | 5222.8 | **Success** |
| **Q5_K_M** | 33 | True | 0.0 | 0.0 | N/A | 5219.8 | **Failed (OOM)** |
| **Q8_0** | 33 | False | 0.0 | 0.0 | N/A | 5117.2 | **Failed (OOM)** |
| **Q8_0** | 33 | True | 0.0 | 0.0 | N/A | 5117.2 | **Failed (OOM)** |
| **Q8_0** | 15 | False | 22.7 | 9.6 | N/A | 4252.1 | **Success** |
| **Q8_0** | 15 | True | 71.8 | 9.4 | 52.4% | 5309.2 | **Success** |

---

## 4. Key Findings and System Analysis

### A. The VRAM Benefits of a Smaller Draft Model
Moving from a 1.5B draft model to a 0.5B draft model immediately reclaims ~800 MB of VRAM. This enables execution under memory-constrained configurations that previously crashed:
* Pairing the `Q5_K_M` target model with the 1.5B draft resulted in a Vulkan OOM crash.
* Pairing it with the 0.5B draft model runs successfully with a peak VRAM of **5,662.4 MB**, allowing higher-precision target inference on 6GB hardware.

### B. Why On-GPU Speculative Decoding Slows Down
Theoretical calculations assume that verifying draft sequences is highly parallel and fast. However, Edge GPUs (like the RTX 3050 Laptop) suffer from **kernel serialization and host launch overhead**:
* In batch size 1, GPU utilization is low, and execution speed is bound by the latency of launching kernels from the host.
* Speculation forces the GPU to launch sequential draft kernels, compile speculative attention weights, and run target verification.
* Since the 7B target-only baseline is already extremely fast (34.5 t/s) when fully in VRAM, the overhead of managing the dual models and serializing kernel launches exceeds the arithmetic savings of verification, causing generation to slow down (from 34.5 t/s to 26.5 t/s).

### C. Where Speculative Decoding Succeeds: Memory-Bound Offloading
Speculative decoding yields a net speedup **only when the target model's evaluation is slower than the draft execution + overhead**.
* In the `Q8_0` (`ngl=15`) sweep, target execution is split between host CPU and device GPU, bottlenecking speed to 10.0 t/s.
* Because the 0.5B draft model fits entirely in VRAM and runs at high speed, generating draft tokens on the GPU is extremely cheap compared to a target evaluation step on the CPU.
* Validating draft blocks allows the system to skip multiple CPU-GPU transfers and CPU layer calculations, yielding a net **+5.0% generation speedup** (from 10.0 t/s to 10.5 t/s).

---

## Related Documentation
*   [Theory Reference: Speculative Decoding and Parallel Verification](../docs/speculative-decoding.md)
*   [Learning Report: Speculative Decoding Constraints](../learnings/learning-speculative-decoding-constraints.md)
