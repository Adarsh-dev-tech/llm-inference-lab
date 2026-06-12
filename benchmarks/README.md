# Benchmark Reports

This directory contains benchmark reports and experiment logs used throughout the project.

The purpose of these reports is to track the impact of various inference optimizations on:

* Throughput
* Latency
* Memory usage
* GPU utilization
* System resource consumption

---

## Benchmark Environment

### Hardware

| Component | Specification              |
| --------- | -------------------------- |
| CPU       | Intel Core i5 (12th Gen)   |
| RAM       | 16 GB DDR5                 |
| GPU       | NVIDIA RTX 3050 Laptop GPU |
| VRAM      | 6 GB                       |

### Software

| Component        | Version   |
| ---------------- | --------- |
| CUDA             | 13.0      |
| Inference Engine | llama.cpp |

### Baseline Model

| Property     | Value               |
| ------------ | ------------------- |
| Model        | Qwen2.5-7B-Instruct |
| Format       | GGUF                |
| Quantization | Q4_K_M              |

---

## Available Reports

| Date       | Report             |
| ---------- | ------------------ |
| 2026-06-12 | Baseline Benchmark |

---

## Future Benchmark Categories

Planned experiments include:

### Quantization

* Q4_K_M
* Q5_K_M
* Q8_0
* FP16

### GPU Offloading

* Partial offloading
* Full offloading
* Layer allocation studies

### Context Scaling

* 4K context
* 8K context
* 16K context
* Long-context behavior

### KV Cache Experiments

* Cache size impact
* Cache quantization
* Memory usage analysis

### Attention Optimizations

* Flash Attention
* PagedAttention
* Alternative attention implementations

### Serving Frameworks

* llama.cpp
* vLLM
* SGLang
* TensorRT-LLM

### Speculative Decoding

* Draft model selection
* Throughput gains
* Latency reduction

---

All benchmark reports should be stored as individual markdown files with the format:

DD-MM-YYYY-experiment-name.md