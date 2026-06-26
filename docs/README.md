# Conceptual Theory References

This directory contains conceptual theory guides and methodological references that explain the engineering, math, and architecture behind LLM inference optimization techniques.

---

## Guide Index and Summaries

### 1. [Transformer Basics & Self-Attention Mechanics](transformer-basics.md)
*   **File**: `transformer-basics.md`
*   **Focus**: An introductory guide explaining the Transformer architecture from absolute first principles. It covers tokenization, high-dimensional vector embeddings, the dot-product self-attention mechanism, projection weights, and matrix operations.

### 2. [Key-Value (KV) Cache Sizing & Memory Layouts](kv-cache.md)
*   **File**: `kv-cache.md`
*   **Focus**: Details the mathematical sizing formulas of Key-Value (KV) caching in autoregressive generation. Explores Multi-Query Attention (MQA), Grouped-Query Attention (GQA), memory bandwidth ceilings (prefill vs. decode bottlenecks), and PagedAttention dynamic mapping layers.

### 3. [FlashAttention Fused Kernels & SRAM Tiling](flash-attention.md)
*   **File**: `flash-attention.md`
*   **Focus**: Explains why standard self-attention is memory-bandwidth bound. Details the systems architecture of FlashAttention, focusing on SRAM tiling, fused GPU kernels, and incremental online softmax calculations that minimize slow High-Bandwidth Memory (HBM) operations.

### 4. [SnapKV Cache Compaction & Attention Sparsity](snapkv.md)
*   **File**: `snapkv.md`
*   **Focus**: Analyzes the mathematical mechanics of the SnapKV algorithm. Explains self-attention sparsity ("heavy hitters" vs. local context) and details how pooling window observations compress the active KV cache dynamically without significant perplexity loss.

### 5. [Speculative Decoding & Parallel Verification](speculative-decoding.md)
*   **File**: `speculative-decoding.md`
*   **Focus**: Explores the mathematics of candidate drafting and parallel verification. Deconstructs the speculative acceptance probability formula, draft model parameter ratio constraints, and the native C++ verification loop implementation in `llama.cpp`.

### 6. [LLM Inference Benchmarking Methodology](benchmarking-methodology.md)
*   **File**: `benchmarking-methodology.md`
*   **Focus**: Establishes local hardware experimental controls for laptop GPUs. Details thermal stabilization (cool-down intervals), subprocess isolation protocols to prevent memory leak pollution, and definitions for telemetry metrics (TTFT, TPS, VRAM, and power draw).

### 7. [Curriculum Tasklist](tasklist.md)
*   **File**: `tasklist.md`
*   **Focus**: A living task tracker cataloging completed and upcoming development phases of the LLM Inference Optimization Lab.

### 8. [Curriculum Roadmap](llm_inference_benchmarking_roadmap.md)
*   **File**: `llm_inference_benchmarking_roadmap.md`
*   **Focus**: A comprehensive roadmap detailing research goals, milestones, and implementation phases across different months of study.
