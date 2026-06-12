# Baseline Benchmark

Date: 12-06-2026

## Objective

Establish baseline inference performance for future optimization experiments.

---

## Hardware

| Component | Specification              |
| --------- | -------------------------- |
| CPU       | Intel Core i5 (12th Gen)   |
| RAM       | 16 GB DDR5                 |
| GPU       | NVIDIA RTX 3050 Laptop GPU |
| VRAM      | 6 GB                       |

---

## Model

| Property     | Value               |
| ------------ | ------------------- |
| Model        | Qwen2.5-7B-Instruct |
| Format       | GGUF                |
| Quantization | Q4_K_M              |

---

## Experiment 1: Partial GPU Offload

### Configuration

```bash
llama-cli \
  -m qwen2.5-7b-instruct-q4_k_m.gguf \
  -ngl 20
```

### Results

| Metric                | Value           |
| --------------------- | --------------- |
| Prompt Throughput     | 35.9 tokens/sec |
| Generation Throughput | 21.1 tokens/sec |
| VRAM Usage            | ~4.16 GB        |

---

## Experiment 2: Full GPU Offload

### Configuration

```bash
llama-cli \
  -m qwen2.5-7b-instruct-q4_k_m.gguf \
  -ngl 33
```

### Results

| Metric                | Value           |
| --------------------- | --------------- |
| Prompt Throughput     | 91.5 tokens/sec |
| Generation Throughput | 35.7 tokens/sec |
| VRAM Usage            | ~4.54 GB        |

---

## Comparison

| Metric                | Partial Offload | Full Offload | Improvement |
| --------------------- | --------------- | ------------ | ----------- |
| Prompt Throughput     | 35.9 t/s        | 91.5 t/s     | +154.9%     |
| Generation Throughput | 21.1 t/s        | 35.7 t/s     | +69.2%      |
| VRAM Usage            | 4.16 GB         | 4.54 GB      | +0.38 GB    |

---

## Conclusion

Full GPU offloading produced significantly higher throughput while requiring only a modest increase in VRAM consumption.

This benchmark serves as the project's baseline reference for all future inference optimization experiments.