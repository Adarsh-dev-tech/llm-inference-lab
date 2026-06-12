# LLM INFERENCE LAB

## Purpose

This project aims to study and optimize Large Language Model (LLM) inference on consumer-grade hardware. The objective is to build a strong understanding of modern inference systems, including quantization, GPU offloading, KV cache management, batching strategies, speculative decoding, and serving frameworks.

To evaluate the impact of these optimizations, a baseline hardware platform and baseline model were selected.

---

# Hardware Specification

## System Configuration

| Component        | Specification                      |
| ---------------- | ---------------------------------- |
| CPU              | Intel Core i5 (12th Generation)    |
| RAM              | 16 GB DDR5                         |
| GPU              | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM             | 6 GB                               |
| OS               | Windows                            |
| CUDA Version     | 13.0                               |
| Inference Engine | llama.cpp                          |

### Hardware Constraints

The system represents a typical mid-range developer laptop rather than a dedicated AI workstation.

Key limitations include:

* Only 6 GB of GPU VRAM
* Limited system memory compared to server-grade machines
* Single consumer GPU
* No multi-GPU support

These constraints make the environment suitable for studying optimization techniques that enable efficient inference on resource-constrained hardware.

---

# Baseline Model Selection

## Chosen Model

**Qwen2.5-7B-Instruct (Q4_K_M GGUF)**

### Quantization

* Format: GGUF
* Quantization: Q4_K_M
* Approximate Size: ~4–5 GB

### Inference Backend

The model is executed using llama.cpp with CUDA acceleration enabled.

Example launch command:

```bash
llama-cli \
  -m qwen2.5-7b-instruct-q4_k_m.gguf \
  -ngl 999
```

---

# Why Qwen2.5-7B Was Chosen

The model was selected because it provides a balance between:

* Model quality
* Hardware feasibility
* Experimental usefulness

### 1. Large Enough To Expose Real Bottlenecks

Smaller models (1B–3B parameters) often run very quickly, making it difficult to observe the impact of inference optimizations.

A 7B model is large enough that:

* GPU offloading matters
* Memory bandwidth becomes relevant
* Quantization effects are measurable
* Throughput improvements are noticeable

This makes it an effective baseline for systems experimentation.

---

### 2. Fits Within Hardware Constraints

The RTX 3050 provides only 6 GB of VRAM.

A Q4_K_M quantized 7B model can be largely or fully offloaded to the GPU while remaining within available memory limits.

Larger models such as 14B, 32B, or 70B would require substantial CPU participation, making it difficult to isolate the effects of GPU-side optimizations.

---

### 3. Representative Of Modern Open Models

The 7B parameter class has become a common benchmark size for:

* Research experiments
* Local inference
* Consumer hardware deployment
* Serving system development

Many inference frameworks evaluate performance using models in this size range.

---

### 4. Suitable For Future Experiments

The model can be reused for studying:

* GPU offloading strategies
* Quantization trade-offs
* KV cache optimization
* Flash Attention
* Continuous batching
* PagedAttention
* Speculative decoding
* Throughput benchmarking
* Latency analysis

Using a fixed baseline model ensures that future performance improvements can be attributed to system-level optimizations rather than model changes.

---

# Benchmark Results

## Baseline Benchmark

The following benchmark was conducted using llama.cpp with CUDA acceleration enabled.

### Partial GPU Offload

Configuration:

```bash
llama-cli \
  -m qwen2.5-7b-instruct-q4_k_m.gguf \
  -ngl 20
```

Results:

| Metric                | Value           |
| --------------------- | --------------- |
| Prompt Throughput     | 35.9 tokens/sec |
| Generation Throughput | 21.1 tokens/sec |
| VRAM Usage            | ~4.16 GB        |

---

### Full GPU Offload

Configuration:

```bash
llama-cli \
  -m qwen2.5-7b-instruct-q4_k_m.gguf \
  -ngl 33
```

Results:

| Metric                | Value           |
| --------------------- | --------------- |
| Prompt Throughput     | 91.5 tokens/sec |
| Generation Throughput | 35.7 tokens/sec |
| VRAM Usage            | ~4.54 GB        |

---

### Observations

Moving from partial to full GPU offloading required approximately 380 MB of additional VRAM but resulted in:

* 2.55× faster prompt processing
* 1.69× faster token generation

This demonstrates the significant performance impact of maximizing GPU residency for transformer layers.

Detailed benchmark reports can be found in the `benchmarks/` directory.

---

# Baseline Summary

| Component             | Value                    |
| --------------------- | ------------------------ |
| CPU                   | Intel Core i5 (12th Gen) |
| RAM                   | 16 GB DDR5               |
| GPU                   | RTX 3050 Laptop GPU      |
| VRAM                  | 6 GB                     |
| Inference Engine      | llama.cpp                |
| Baseline Model        | Qwen2.5-7B-Instruct      |
| Quantization          | Q4_K_M                   |
| GPU Offload           | Full (`-ngl 33`)         |
| VRAM Usage            | ~4.54 GB                 |
| Prompt Throughput     | 91.5 tokens/sec          |
| Generation Throughput | 35.7 tokens/sec          |

This configuration serves as the reference point for all future optimization experiments.

---

# LLM Inference Benchmark Framework

This project includes a production-grade Python benchmarking framework designed to run, measure, and record local LLM inference performance. It provides modular separation of concerns, high-precision C++ level timings directly from the `llama.cpp` runtime, comprehensive system hardware telemetry (VRAM, RAM, CPU/GPU utilization, GPU clocks, power draw, and temperatures), multi-iteration testing with statistical aggregations, and dual-mode data persistence.

## Project Structure

```text
llm-inference-lab/
│
├── benchmark.py            # Main CLI benchmark runner and orchestrator
├── requirements.txt        # Package dependencies
├── README.md               # Project documentation (this file)
│
├── prompts/                # Test prompts
│   ├── short.txt
│   ├── medium.txt
│   └── long.txt
│
├── results/                # Output storage
│   ├── benchmark_history.csv    # Master CSV experiment database (append-only)
│   └── json/                    # Per-run detailed JSON reports
│       └── YYYY-MM-DD_HH-MM-SS.json
│
└── utils/                  # Core modules
    ├── system_monitor.py   # Telemetry thread (CPU, GPU, clocks, power, temp)
    ├── metrics.py          # Latency and throughput math
    └── logging.py          # CSV and JSON database writers
```

## Setup & Installation

1. Create a local virtual environment:
   ```powershell
   uv venv --python C:\Users\adars\AppData\Local\Programs\Python\Python312\python.exe
   .venv\Scripts\Activate.ps1
   ```

2. Compile and install `llama-cpp-python` with CUDA acceleration enabled on Windows:
   ```powershell
   $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
   $env:CudaToolkitDir = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
   $env:CMAKE_ARGS = "-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe'"
   $env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;" + $env:PATH
   uv pip install -r requirements.txt --no-binary llama-cpp-python --no-cache
   ```

## CLI Usage

Run a benchmark with the following command (make sure the CUDA DLL directory is added to your path on Windows so dependencies load correctly):

```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;" + $env:PATH
python benchmark.py `
  --model C:\Users\adars\Downloads\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf `
  --quant Q4_K_M `
  --prompt prompts/medium.txt `
  --ngl 33 `
  --max_tokens 200 `
  --iterations 5
```

### Options:
* `--model`: Path to GGUF model file.
* `--quant`: Quantization tag (e.g., Q4_K_M).
* `--prompt`: Path to prompt text file.
* `--ngl`: Number of layers to offload to GPU (default `33`).
* `--max_tokens`: Max output length (default `200`).
* `--ctx`: Context window size (default `4096`).
* `--temp`: Temperature (default `0.7`).
* `--rep_pen`: Repetition penalty (default `1.1`).
* `--iterations`: Number of times to run the benchmark to collect mean, standard deviation, and p95 percentiles (default `1`).

## Data Exporters & Schema

The framework records every run in two locations:

### 1. Historical Dataset (`results/benchmark_history.csv`)
An append-only database recording each individual iteration. It contains the following columns:

| Column Name | Description |
|---|---|
| `run_id` | Unique ID generated per benchmark session (`RUN-YYYYMMDD-XXXXXX`) |
| `iteration_index` | 1-based index representing the repetition run |
| `timestamp` | Date and time in `YYYY-MM-DDTHH:MM:SS` format |
| `model` | Name of the evaluated model file |
| `quantization` | Quantization type (e.g. `Q4_K_M`) |
| `n_gpu_layers` | Layers offloaded to the GPU |
| `prompt_tokens` | Exact count of tokens in prompt |
| `generated_tokens` | Exact count of tokens generated (calculated via C++ API/tokenizer) |
| `prompt_tps` | Prompt evaluation throughput (tokens/sec) using C++ prefill timing |
| `generation_tps` | Generation throughput (tokens/sec) using C++ decode timing |
| `first_token_latency_ms` | Monotonic clock latency to first token (TTFT) in ms |
| `total_request_latency_ms` | Total elapsed request time in ms |
| `ram_before_mb` | System-wide RAM used before generation |
| `ram_after_mb` | System-wide RAM used after generation |
| `vram_before_mb` | System-wide VRAM used before generation |
| `vram_after_mb` | System-wide VRAM used after generation |
| `context_length` | Context window constraint size |
| `total_generation_time_ms` | Monotonic clock decode duration in ms |
| `cpu_utilization` | Average CPU usage % during generation |
| `gpu_utilization` | Average GPU kernel execution % during generation |
| `avg_itl_ms` | Average Inter-Token Latency (ITL) in ms per token |
| `process_ram_before_mb` | Resident Set Size (RSS) of Python process before gen |
| `process_ram_after_mb` | Resident Set Size (RSS) of Python process after gen |
| `gpu_mem_utilization` | Average GPU memory controller utilization % |
| `gpu_temperature_c` | Average GPU temperature in °C |
| `gpu_power_watts` | Average GPU board power draw in Watts |
| `gpu_graphics_clock_mhz` | Average GPU Core clock speed in MHz |
| `gpu_memory_clock_mhz` | Average GPU Memory clock speed in MHz |
| `ram_before_load_mb` | System RAM usage before model load |
| `ram_after_load_mb` | System RAM usage after model load (model memory footprint) |
| `vram_before_load_mb` | System VRAM usage before model load |
| `vram_after_load_mb` | System VRAM usage after model load (model VRAM footprint) |

---

### 2. JSON Artifacts (`results/json/`)
A structured snapshot report is written for each benchmark session, containing aggregated statistics (mean, standard deviation, min, max, median, p95) across all iterations, system hardware information, configuration parameters, and detailed metric logs.