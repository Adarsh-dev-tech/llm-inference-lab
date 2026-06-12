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