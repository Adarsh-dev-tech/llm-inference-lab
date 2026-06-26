# Benchmarking Results and Telemetry Databases

This directory stores the saved results, benchmark sweeps databases (CSV), and fine-grained, high-frequency execution telemetry reports (JSON) compiled across the optimization stages.

---

## Directory Structure

```text
results/
│
├── benchmark_history.csv             # Baseline & layer offloading runs
├── flash_attn_sweeps.csv             # FlashAttention vs. Standard Attention results
├── kv_cache_growth.csv               # Context length scaling and VRAM consumption records
├── paged_attention_sweeps.csv        # Prefix caching speedup metrics
├── snapkv_benchmark.csv              # SnapKV perplexity and VRAM sweeps
├── speculative_decoding_benchmark.csv # Speculative decoding speedups and acceptance rates
│
└── json/                             # Raw high-frequency telemetry & sweep config outputs
```

---

## Tabular CSV Schemas

### 1. `benchmark_history.csv`
Tracks baseline token generation performance and system metrics.
*   `run_id`: Unique identifier for the benchmark session.
*   `iteration_index`: 1-based index representing the repetition run.
*   `timestamp`: Execution date and time.
*   `model`: Target model file name.
*   `quantization`: Model weight bit-width (e.g., `Q4_K_M`, `Q5_K_M`, `Q8_0`).
*   `n_gpu_layers`: Number of layers offloaded to the GPU.
*   `prompt_tokens`: Count of input prompt tokens.
*   `generated_tokens`: Count of output tokens.
*   `prompt_tps`: Prompt processing throughput (tokens/sec).
*   `generation_tps`: Token generation throughput (tokens/sec).
*   `first_token_latency_ms`: Time to first token (TTFT) in milliseconds.
*   `total_request_latency_ms`: Total execution time in milliseconds.
*   `ram_before_mb` / `ram_after_mb`: System RAM utilization before and after generation.
*   `vram_before_mb` / `vram_after_mb`: GPU VRAM utilization before and after generation.
*   `context_length`: Active context window size.
*   `total_generation_time_ms`: Time spent in autoregressive generation.
*   `cpu_utilization`: System CPU utilization percentage.
*   `gpu_utilization`: GPU kernel execution percentage.
*   `avg_itl_ms`: Average inter-token latency (ITL) in milliseconds.
*   `gpu_power_watts`: Average board power draw in Watts.
*   `gpu_graphics_clock_mhz`: Average GPU core clock frequency.

### 2. `flash_attn_sweeps.csv`
Compares FlashAttention performance against Standard Attention.
*   `timestamp`: Run date and time.
*   `quantization`: Weight bit-width.
*   `ngl`: Number of GPU-offloaded layers.
*   `flash_attn`: Boolean indicating if FlashAttention is enabled.
*   `context_length`: Size of the context sequence sweep.
*   `ambient_vram_mb`: GPU VRAM in idle state prior to loading.
*   `post_load_vram_mb`: VRAM occupied after loading weights.
*   `prefill_peak_vram_mb`: Peak VRAM recorded during parallel prefill.
*   `prefill_spike_mb`: Memory allocation overhead above baseline.
*   `post_decode_vram_mb`: VRAM level after finishing generation.
*   `prefill_time_sec`: Time elapsed during prompt processing.
*   `decode_speed_tps`: Autoregressive throughput (tokens/sec).
*   `is_paging_triggered`: Indicator of physical VRAM overflow onto system RAM (PCIe swapping).

### 3. `kv_cache_growth.csv`
Records memory footprint growth across context sequence limits.
*   `timestamp`: Sweep run timestamp.
*   `quantization`: Weight quantization.
*   `context_length`: Sweep length (128 to 4096 tokens).
*   `ngl`: GPU layers.
*   `theoretical_kv_mb`: Mathematically calculated KV cache size in MB.
*   `ambient_vram_mb` / `post_load_vram_mb`: Pre-load and post-load baseline VRAM.
*   `prefill_peak_vram_mb` / `prefill_spike_mb`: Peak usage and delta during prefill.
*   `experimental_kv_estimate_mb`: VRAM growth isolated to the KV cache footprint.
*   `is_paging_triggered`: PCIe RAM paging warning.
*   `load_time_sec` / `prefill_time_sec` / `decode_step_time_sec`: Execution timing metrics.

### 4. `paged_attention_sweeps.csv`
Details prompt caching latency speedups.
*   `timestamp`: Measurement timestamp.
*   `quantization`: Weight quantization.
*   `ngl`: GPU-offloaded layers.
*   `prompt_cache_enabled`: Boolean flag for prefix caching.
*   `ambient_vram_mb` / `post_load_vram_mb`: Pre-load and post-load baseline VRAM.
*   `cold_prefill_time_sec`: Prefill time on a cold start (cache miss).
*   `cold_prefill_vram_mb`: VRAM occupied during cold prefill.
*   `warm_prefill_time_sec`: Prefill time on a warm start (cache hit).
*   `warm_prefill_vram_mb`: VRAM occupied during warm prefill.
*   `prefill_latency_speedup_x`: Latency reduction ratio (cold / warm).

### 5. `snapkv_benchmark.csv`
Compiles perplexity and memory compression profiles of SnapKV.
*   `timestamp`: Sweep run timestamp.
*   `model`: Base model identifier.
*   `precision_bits`: Weight bit-width (4-bit NF4 / 8-bit Int8).
*   `k_val`: Key-Value token retention limit ($K \in \{16, 32, 64, 128, 256\}$).
*   `observation_window`: Prefix pooling window length ($L_{\text{obs}}$).
*   `recent_window`: Local suffix retention window ($L_{\text{rec}}$).
*   `perplexity`: Evaluated text generation perplexity on WikiText-2.
*   `loaded_vram_mb`: Baseline memory occupied by the model weights.
*   `prefill_peak_vram_mb`: Peak VRAM recorded during SnapKV compression prefill.

### 6. `speculative_decoding_benchmark.csv`
Measures speculative speedups and draft model acceptance.
*   `timestamp`: Evaluation timestamp.
*   `target_quantization`: Quantization of the larger target model.
*   `target_ngl`: GPU layers of the target model.
*   `speculative_enabled`: Boolean flag indicating if speculative decoding was active.
*   `draft_model`: Path/name of the draft GGUF model file.
*   `prompt_tps` / `generation_tps`: Prefill and decode throughput (tokens/sec).
*   `acceptance_rate`: Percentage of candidate tokens accepted by the target model.
*   `peak_vram_mb`: Peak VRAM consumed by both models and active caches.

---

## JSON Telemetry Reports

The `json/` directory contains complete, granular telemetry logs. These files are committed to document high-frequency hardware metrics and verification traces:
1.  **Detailed Run Logs (`YYYY-MM-DD_HH-MM-SS.json`)**:
    *   Record complete hardware specs, precise generation parameters, and aggregated statistics (mean, stdev, median, p95) of latency metrics.
    *   Contain raw timeseries samples logging CPU load, GPU kernel utilization, GPU clock frequencies, power draw in Watts, and temperatures at 50 Hz intervals.
2.  **Profiler Sweeps Metadata (`<optimization>_profile_*.json`)**:
    *   Stores serialized JSON sweeps outputs containing parameters and results (e.g., SnapKV perplexity matrices, KV cache growth rates) that back up the markdown benchmarks.
