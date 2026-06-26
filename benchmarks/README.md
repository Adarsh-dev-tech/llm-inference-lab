# Empirical Benchmark Reports

This directory compiles empirical benchmark reports and profiling sweeps for LLM inference on consumer-grade hardware. Each report analyzes the impact of specific optimization techniques on speed (throughput in tokens/sec), response latency (time to first token, inter-token latency), and system-level boundaries (VRAM/RAM consumption, GPU power draw, and temperature limits).

---

## Hardware Profile
All benchmark runs are executed on a standard consumer laptop environment representing localized hardware bottlenecks:
*   **CPU**: Intel Core i5 (12th Generation)
*   **System RAM**: 16 GB DDR5
*   **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM, 128-bit bus)
*   **OS**: Windows 11

---

## Summary of Completed Benchmark Reports

### 1. [Baseline Benchmark](12-06-2026-baseline-benchmark.md)
*   **File**: `12-06-2026-baseline-benchmark.md`
*   **Focus**: Establishes baseline inference performance (speed, latency, resource usage) of the `Qwen2.5-7B-Instruct` GGUF model without optimization. It validates standard throughput boundaries on CPU and partial GPU offloads, setting the reference mark for subsequent optimization sweeps.

### 2. [Quantization & GPU Residency Sweeps](15-06-2026-quantization-benchmark.md)
*   **File**: `15-06-2026-quantization-benchmark.md`
*   **Focus**: Evaluates GGUF quantization formats (`Q4_K_M`, `Q5_K_M`, and `Q8_0`) under a strict 6 GB physical VRAM budget. Identifies the "performance cliff" (PCIe-RAM swapping slowdown) that occurs when model footprint exceeds physical VRAM limits.

### 3. [KV Cache Memory Profiler](23-06-2026-kv-cache-profiler.md)
*   **File**: `23-06-2026-kv-cache-profiler.md`
*   **Focus**: Analyzes peak VRAM consumption of static model weights combined with dynamic Key-Value (KV) cache sizes across context lengths ($128$ to $4096$ tokens). Isolates memory leaks and verifies the mathematical scaling constraints of multi-layer attention matrices.

### 4. [FlashAttention Performance Sweeps](25-06-2026-flash-attention-benchmark.md)
*   **File**: `25-06-2026-flash-attention-benchmark.md`
*   **Focus**: Compares prefill duration and memory footprints when FlashAttention is enabled (`flash_attn=True`) versus standard attention. Documents the arithmetic intensity gains and SRAM tiling savings that accelerate prompt processing times.

### 5. [SnapKV Cache Compaction Evaluator](26-06-2026-snapkv-compression.md)
*   **File**: `26-06-2026-snapkv-compression.md`
*   **Focus**: Evaluates the perplexity impact and VRAM savings of the SnapKV pooling window clustering algorithm. Tests actual weight quantization formats (`4-bit NF4` and `8-bit Int8` loaded via `bitsandbytes`) across token retention budgets $K \in \{16, 32, 64, 128, 256\}$.

### 6. [PagedAttention & Prefix Caching Benchmark](27-06-2026-paged-attention-benchmark.md)
*   **File**: `27-06-2026-paged-attention-benchmark.md`
*   **Focus**: Benchmarks prompt prefix caching (shared pages) in `llama-cpp-python`. Compares warm cache hits vs. cold starts across quantization profiles, highlighting the 35x-37x prefill speedups achieved by caching static contexts.

### 7. [Native Speculative Decoding Sweeps](28-06-2026-speculative-decoding.md)
*   **File**: `28-06-2026-speculative-decoding.md`
*   **Focus**: Profiles native speculative decoding in `llama.cpp` using a `Qwen2.5-7B` target model paired with `0.5B` and `1.5B` draft models. Analyzes acceptance rates, draft evaluation overhead, GPU serialization bottlenecks, and the 5% speedup achieved during CPU-split offloads.