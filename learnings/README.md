# Post-Mortems and Lessons Learned

This directory indexes learnings, hardware observations, compilation walkthroughs, and code-level gotchas compiled after each optimization phase.

---

## Learnings Index and Summaries

### 1. [Hardware Setup, Compilation & Layer Offloading](learning-hardware-setup.md)
*   **File**: `learning-hardware-setup.md`
*   **Focus**: Logs the installation and source compilation steps of `llama.cpp` with CUDA support on Windows. Captures system-level gotchas regarding PATH variable setups and layer-offloading boundaries.

### 2. [Quantization Trade-Offs & Memory Bandwidth Limits](learning-quantization-differences.md)
*   **File**: `learning-quantization-differences.md`
*   **Focus**: Analyzes the VRAM headroom margins and perplexity trade-offs between `Q4_K_M`, `Q5_K_M`, and `Q8_0` GGUF models. Connects weight sizes to KV cache capacity buffers.

### 3. [KV Cache Math, Arithmetic Intensity & Hardware Limits](learning-kv-cache-math.md)
*   **File**: `learning-kv-cache-math.md`
*   **Focus**: Explains the shifts in hardware bounds between prefill (compute-bound) and decode (memory-bandwidth bound) stages. Outlines GQA compression ratios and advanced memory-saving strategies like PagedAttention.

### 4. [FlashAttention Memory I/O, Arithmetic Intensity & GPU Architecture](learning-flashattention-memory-io.md)
*   **File**: `learning-flashattention-memory-io.md`
*   **Focus**: Deconstructs modern GPU memory latency hierarchies (DRAM, SRAM, registers, cache-lines). Illustrates mathematically how FlashAttention tiles and fuses kernels to reduce memory reads and writes.

### 5. [SnapKV Cache Compaction Mechanics](learning-snapkv-mechanics.md)
*   **File**: `learning-snapkv-mechanics.md`
*   **Focus**: Logs post-mortem findings of attention-pooling compression. Focuses on position ID alignment bugs during Rotary Position Embedding (RoPE) calculations, and clustering threshold behaviors.

### 6. [Native Speculative Decoding Constraints](learning-speculative-decoding-constraints.md)
*   **File**: `learning-speculative-decoding-constraints.md`
*   **Focus**: Discusses edge GPU serialization bottlenecks (kernel queuing latency) that slow down native speculation during full GPU offloads, and details when CPU-split offloads yield positive speedups.

### 7. [Transformer Weight Distribution & Context Limits](learning-transformer-internals.md)
*   **File**: `learning-transformer-internals.md`
*   **Focus**: Deconstructs Qwen2.5 parameter allocations across Self-Attention projections, Feed-Forward Network (FFN/MLP) blocks, and embedding layers. Maps mathematical VRAM growth profiles.
