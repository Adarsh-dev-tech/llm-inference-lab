# Theory Reference: Key-Value (KV) Cache Sizing and Memory Layouts

In autoregressive Large Language Model (LLM) generation, tokens are predicted one at a time. To predict token $t$, the self-attention layer must calculate dot-products between the Query vector of token $t$ ($q_t$) and the Key vectors of all previous tokens ($K_{1 \dots t}$), and then compute a weighted sum over the Value vectors ($V_{1 \dots t}$).

To avoid recalculating the Key ($K$) and Value ($V$) vectors of all historical tokens at every single generation step, these vectors are cached in VRAM. This is known as **Key-Value (KV) Caching**.

---

## 1. KV Cache Sizing Mathematics

The memory footprint of the KV cache is determined by the model's depth, head configuration, sequence length, batch size, and numerical precision format.

### Sizing Equation (Bytes)

$$\text{KV Cache Size (bytes)} = 2 \times L \times H_{\text{kv}} \times D \times \text{BPE} \times T \times B$$

Where:
*   **$2$**: Factor representing that we store both **Key** ($K$) and **Value** ($V$) vectors.
*   **$L$**: Number of Transformer layers in the model (e.g., $28$ for Qwen2.5-7B).
*   **$H_{\text{kv}}$**: Number of Key/Value heads per layer.
*   **$D$**: Head dimension (size of each head, e.g., $128$).
*   **$\text{BPE}$**: Bytes Per Element based on numerical precision (e.g., $2$ for FP16 / BF16).
*   **$T$**: Context sequence length (number of tokens currently in the cache).
*   **$B$**: Batch size (number of parallel sequences being generated, $B=1$ for local single-user inference).

---

## 2. Attention Mechanics & KV Head Configurations

The size of the KV cache is highly dependent on how keys and values are distributed across heads.

```
Multi-Head Attention (MHA)       Grouped-Query Attention (GQA)       Multi-Query Attention (MQA)
    Q Heads    K/V Heads             Q Heads     K/V Heads             Q Heads    K/V Head
    +---+      +---+                 +---+---+   +---+                 +---+---+  +---+
    | Q | ---> | K |                 | Q | Q | \ |   |                 | Q | Q |  |   |
    +---+      +---+                 +---+---+  >| K |                 +---+---+  | K |
    | Q | ---> | K |                 | Q | Q | / |   |                 | Q | Q |  |   |
    +---+      +---+                 +---+---+   +---+                 +---+---+  +---+
```

### A. Multi-Head Attention (MHA)
*   **Mechanism**: Every Query head has its own corresponding Key and Value head ($H_{\text{kv}} = H_{\text{q}}$).
*   **Qwen2.5-7B theoretical MHA size** (if it did not use GQA):
    *   $H_{\text{kv}} = H_{\text{q}} = 28$
    *   $\text{Size per token} = 2 \times 28 \times 28 \times 128 \times 2 = 401,408 \text{ bytes} \approx \mathbf{401.4\text{ KB/token}}$
    *   At **4,096 tokens**: $1.64\text{ GB}$ VRAM cache.

### B. Grouped-Query Attention (GQA)
*   **Mechanism**: Query heads are grouped into clusters, and each cluster shares a single Key/Value head ($H_{\text{kv}} < H_{\text{q}}$). In Qwen2.5-7B, the query-to-KV ratio is $7:1$ ($28$ query heads share $4$ KV heads).
*   **Qwen2.5-7B GQA size**:
    *   $H_{\text{kv}} = 4$
    *   $\text{Size per token} = 2 \times 28 \times 4 \times 128 \times 2 = 57,344 \text{ bytes} \approx \mathbf{57.34\text{ KB/token}}$
    *   At **4,096 tokens**: $\approx \mathbf{234.9\text{ MB}}$ VRAM cache (**7x memory reduction** vs. MHA).

### C. Multi-Query Attention (MQA)
*   **Mechanism**: All Query heads share a single Key and Value head ($H_{\text{kv}} = 1$).
*   **Qwen2.5-7B theoretical MQA size**:
    *   $H_{\text{kv}} = 1$
    *   $\text{Size per token} = 2 \times 28 \times 1 \times 128 \times 2 = 14,336 \text{ bytes} \approx \mathbf{14.34\text{ KB/token}}$
    *   At **4,096 tokens**: $\approx \mathbf{58.7\text{ MB}}$ VRAM cache (**28x memory reduction** vs. MHA).

---

## 3. KV Cache Quantization and Precision Formats

The numerical precision format determines the $\text{BPE}$ (Bytes Per Element) multiplier:

| Precision Format | Bits per Weight | BPE Multiplier | KV Cache Size at 4,096 Context (Qwen2.5-7B, GQA) |
| :--- | :--- | :--- | :--- |
| **FP32 (Float32)** | 32 bits | $4$ bytes | $469.8\text{ MB}$ |
| **FP16 / BF16 (Default)** | 16 bits | $2$ bytes | $234.9\text{ MB}$ |
| **Q8_0 (8-bit Quantized)** | 8 bits | $1$ byte | $117.4\text{ MB}$ |
| **Q4_0 (4-bit Quantized)** | 4 bits | $0.5$ bytes | $58.7\text{ MB}$ |

> [!TIP]
> **Cache Quantization**: Quantizing the KV cache (e.g., using 8-bit or 4-bit integers instead of Float16) is a powerful way to reclaim VRAM headroom in memory-constrained environments. For example, moving from standard FP16 KV cache to Q4_0 KV cache halves the VRAM footprint at high context windows, allowing context lengths to scale twice as high on a 6 GB card.

---

## 4. Allocation Logic in llama.cpp

In `llama.cpp` (and the `llama-cpp-python` bindings), the KV cache behaves differently than in native PyTorch implementations:

### Static Pre-Allocation
Unlike dynamic frameworks that allocate memory on the fly as the sequence grows, `llama.cpp` **pre-allocates the entire KV cache buffer statically at model loading time**. 
*   When you initialize `Llama(..., n_ctx=4096)`, llama.cpp allocates a contiguous block of VRAM large enough to hold all 4,096 token slots.
*   This is why VRAM usage spikes immediately after loading the model, even before a single prompt token is processed. 
*   **Advantage**: Prevents Out-Of-Memory (OOM) errors from occurring midway through generation (since the space is reserved upfront).
*   **Disadvantage**: Wastes VRAM if the actual sequence generated is much shorter than `n_ctx`.

### Memory Layout
Inside the GGUF backend, the KV cache is stored as two giant contiguous tensors ($K$-cache and $V$-cache) structured as:

$$\text{Shape}(K) = [L, B, H_{\text{kv}}, T_{\text{max}}, D]$$
$$\text{Shape}(V) = [L, B, H_{\text{kv}}, T_{\text{max}}, D]$$

Where $T_{\text{max}}$ is the maximum context length (`n_ctx`). For Qwen2.5-7B with GQA:
*   The $K$-tensor has shape $[28, 1, 4, 4096, 128]$.
*   For each layer, the key vectors are stored sequentially. As new tokens are processed, llama.cpp indexes into the active sequence slot ($t$) and overwrites the pre-allocated slice for that token.

---

## 5. PagedAttention Memory Address Mapping and Scheduling

To resolve the fragmentation and static allocation issues of `llama.cpp`'s default behavior, modern serving frameworks (such as `vLLM`) use **PagedAttention**. This mechanism separates the logical sequence from physical VRAM by mimicking the page table architecture of operating system virtual memory.

### Logical vs. Physical Mapping
*   **Logical Blocks**: The KV cache for a sequence is represented as a contiguous stream of tokens divided into blocks of size $B_{\text{size}}$ (typically $16$ tokens per block).
*   **Physical Blocks**: Contiguous chunks of VRAM pre-allocated globally. These blocks do not need to be contiguous in physical memory.
*   **Page Table**: A lookup registry that dynamically maps the logical block index to its physical block location during inference.

### Page Address Mapping Scheduler Grid (ASCII)

Here is how PagedAttention maps two parallel sequence prompts that share the same initial system prompt (32 tokens) but diverge during generation:

```
=== LOGICAL SEQUENCE ADDRESS STACKS ===

Sequence 1 (Context: System Prompt [32 tokens] + "A...")
  Logical Blocks:  [ LBN 0 (0-15) ] -> [ LBN 1 (16-31) ] -> [ LBN 2 (32-47) ]
  
Sequence 2 (Context: System Prompt [32 tokens] + "B...")
  Logical Blocks:  [ LBN 0 (0-15) ] -> [ LBN 1 (16-31) ] -> [ LBN 2 (32-47) ]


=== PAGE TABLE MAPS ===

Sequence 1 Page Table:
  +----------------------+-----------------------+
  | Logical Block Number | Physical Block Number |
  +----------------------+-----------------------+
  |        LBN 0         |        PBN 102        |  <-- Shared System Prompt Block 1
  |        LBN 1         |        PBN 103        |  <-- Shared System Prompt Block 2
  |        LBN 2 (Seq 1) |        PBN 501        |  <-- Unique Sequence 1 Generation
  +----------------------+-----------------------+

Sequence 2 Page Table:
  +----------------------+-----------------------+
  | Logical Block Number | Physical Block Number |
  +----------------------+-----------------------+
  |        LBN 0         |        PBN 102        |  <-- Shared System Prompt Block 1
  |        LBN 1         |        PBN 103        |  <-- Shared System Prompt Block 2
  |        LBN 2 (Seq 2) |        PBN 742        |  <-- Unique Sequence 2 Generation
  +----------------------+-----------------------+


=== PHYSICAL VRAM BLOCK ALLOCATION GRID ===

  Physical Block Registry (VRAM memory pool):
  +-----------------------------------------------------------------+
  | ...                                                             |
  | Block 102 (PBN 102) -> [Keys & Values for Tokens 0 - 15]        | <-- Seq 1 & 2 Read-Only
  | Block 103 (PBN 103) -> [Keys & Values for Tokens 16 - 31]       | <-- Seq 1 & 2 Read-Only
  | ...                                                             |
  | Block 501 (PBN 501) -> [Keys & Values for Seq 1 Tokens 32 - 47] | <-- Seq 1 Private Write
  | ...                                                             |
  | Block 742 (PBN 742) -> [Keys & Values for Seq 2 Tokens 32 - 47] | <-- Seq 2 Private Write
  | ...                                                             |
  +-----------------------------------------------------------------+
```

### The Copy-on-Write (CoW) Scheduling Logic
1.  **Shared Prefixes**: When both Sequence 1 and Sequence 2 are initialized, the scheduler assigns them the same physical blocks (`PBN 102` and `PBN 103`) containing the pre-evaluated system prompt. VRAM usage for the prompt is cut exactly in half because only one copy is stored physically.
2.  **Divergent Paths**: The moment Sequence 1 generates its next token at logical index 32, it requires writing new keys and values. Since `PBN 103` is marked as shared (ref count > 1), the scheduler allocates a new physical block (`PBN 501`) exclusively for Sequence 1. 
3.  **Physical Offset Computation**:
    When a GPU thread needs to retrieve the key vector $K_{t}$ for token index $t=35$:
    *   **Block Index**: $t / B_{\text{size}} = 35 / 16 = \mathbf{2}$ (Logical Block $2$).
    *   **Block Offset**: $t \pmod{B_{\text{size}}} = 35 \pmod{16} = \mathbf{3}$ (Slot $3$ within the block).
    *   **Physical Lookup**: Query the Page Table for Logical Block $2 \rightarrow$ returns `PBN 501`.
    *   **Memory Fetch**: Fetch vector at address:
        $$\text{Address} = \text{BaseAddress}(\text{PBN } 501) + (3 \times D \times \text{BPE})$$
        Where $D$ is the head dimension and $\text{BPE}$ is the bytes per element.

---

## Related Documentation
*   [Theory Reference: Transformer Architectures and Self-Attention Mechanics](transformer-basics.md)
*   [Theory Reference: FlashAttention Fused Kernels and SRAM Tiling](flash-attention.md)
*   [Theory Reference: SnapKV Compression and Attention Sparsity](snapkv.md)
*   [Benchmark Report: KV Cache Memory Profile on Qwen models](../benchmarks/23-06-2026-kv-cache-profiler.md)
