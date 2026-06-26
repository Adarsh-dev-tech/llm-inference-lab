# Learning Report: FlashAttention Memory I/O, Arithmetic Intensity, and GPU Architecture

This report analyzes the hardware-level constraints of self-attention, explaining why standard attention scales poorly on GPU hardware. Using the concepts of **Arithmetic Intensity** and the **Roofline Model**, we mathematically demonstrate how FlashAttention bypasses High-Bandwidth Memory (HBM) bandwidth limits by leveraging GPU SRAM and register-level kernel fusion.

---

## 1. GPU Memory Hierarchy and Performance Specs

To understand why attention is slow, we must analyze the hardware interface of modern GPUs (such as the NVIDIA RTX 3050 Laptop GPU used in our benchmarks).

Modern GPUs consist of multiple **Streaming Multiprocessors (SMs)**. Each SM has high-speed execution units (CUDA cores, Tensor cores) and a hierarchy of memory storage:

```
+---------------------------+-------------------+----------------------+-------------------+
| Memory Type               | Location          | Typical Bandwidth    | Typical Latency   |
+---------------------------+-------------------+----------------------+-------------------+
| Registers                 | On-SM (Per-core)  | ~30 - 80 TB/s        | 1 cycle           |
| SRAM (Shared Memory)      | On-SM (Per-SM)    | ~19 TB/s             | ~20 - 30 cycles   |
| L2 Cache                  | On-Chip (Shared)  | ~3 - 5 TB/s          | ~100 - 200 cycles |
| HBM (VRAM / Global Mem)   | Off-Chip (DRAM)   | ~0.2 - 2.0 TB/s      | ~400 - 800 cycles |
+---------------------------+-------------------+----------------------+-------------------+
```

### The Performance Gap
A standard consumer GPU can execute floating-point math at massive speeds:
*   An RTX 3050 Laptop GPU has a peak half-precision tensor compute limit of **$\approx 36$ TFLOPS** ($3.6 \times 10^{13}$ Floating Point Operations per second).
*   However, its VRAM (HBM) memory bandwidth is only **$\approx 192$ GB/s** ($1.92 \times 10^{11}$ bytes per second).
*   **The Ratio**: The compute-to-memory-bandwidth ratio is:
    $$\frac{36 \times 10^{12} \text{ FLOPS}}{192 \times 10^{9} \text{ Bytes/s}} = \mathbf{187.5 \text{ FLOPs per Byte}}$$
*   **System Meaning**: If a program does not perform at least **$187.5$ math operations (FLOPs)** for every **$1$ byte** of data it reads from or writes to VRAM, the GPU cores will sit idle waiting for memory. The program is **memory-bandwidth bound**.

---

## 2. Arithmetic Intensity of Self-Attention

Let's calculate the **Arithmetic Intensity** ($\text{FLOPs}/\text{Byte}$) of a self-attention layer to see where it falls on the hardware spectrum.

Given:
*   Sequence length $T$
*   Head dimension $D$ (typically $128$)
*   FP16 precision ($2$ bytes per element)

During prompt evaluation (prefill), the attention matrix calculation is:
$$S = Q K^T$$

### A. Memory Transfer (Bytes Read and Written to HBM)
In standard attention, we must transfer the following matrices to and from HBM:
1.  **Read** $Q$ of size $[T, D]$: $2 \cdot T \cdot D$ bytes.
2.  **Read** $K$ of size $[T, D]$: $2 \cdot T \cdot D$ bytes.
3.  **Write** $S$ of size $[T, T]$ to HBM: $2 \cdot T^2$ bytes.
4.  **Read** $S$ from HBM to compute softmax: $2 \cdot T^2$ bytes.
5.  **Write** $P = \text{softmax}(S)$ of size $[T, T]$ to HBM: $2 \cdot T^2$ bytes.
6.  **Read** $P$ of size $[T, T]$ to multiply by $V$: $2 \cdot T^2$ bytes.
7.  **Read** $V$ of size $[T, D]$: $2 \cdot T \cdot D$ bytes.
8.  **Write** Output $O$ of size $[T, D]$: $2 \cdot T \cdot D$ bytes.

Adding these up:
$$\text{Total HBM Bytes} = 8 \cdot T \cdot D + 8 \cdot T^2$$

### B. Computational Work (FLOPs Performed)
For matrix operations:
1.  **$Q K^T$ Matrix Multiplication**: Shapes $[T, D] \times [D, T]$ requires $2 \cdot T^2 \cdot D$ FLOPs (each multiply-accumulate is 2 operations).
2.  **Softmax**: Computing max, subtract, exponentiate, and divide takes $\approx 3$ FLOPs per element: $3 \cdot T^2$ FLOPs.
3.  **$P V$ Matrix Multiplication**: Shapes $[T, T] \times [T, D]$ requires $2 \cdot T^2 \cdot D$ FLOPs.

Adding these up:
$$\text{Total Compute FLOPs} = 4 \cdot T^2 \cdot D + 3 \cdot T^2 = T^2 (4D + 3)$$

### C. Arithmetic Intensity Equation
$$\text{Arithmetic Intensity} = \frac{\text{Total Compute FLOPs}}{\text{Total HBM Bytes}} = \frac{T^2 (4D + 3)}{8 \cdot T \cdot D + 8 \cdot T^2}$$

Dividing numerator and denominator by $T^2$:
$$\text{Arithmetic Intensity} = \frac{4D + 3}{8 \cdot (D/T) + 8}$$

### D. Case Study: Small vs. Large Context Lengths
Let's plug in $D = 128$ for two context lengths:

#### 1. Low Context ($T = 512$)
$$\text{Arithmetic Intensity} = \frac{4(128) + 3}{8 \cdot (128/512) + 8} = \frac{515}{2 + 8} = \mathbf{51.5 \text{ FLOPs/Byte}}$$

#### 2. High Context ($T = 4096$)
$$\text{Arithmetic Intensity} = \frac{4(128) + 3}{8 \cdot (128/4096) + 8} = \frac{515}{0.25 + 8} = \frac{515}{8.25} = \mathbf{62.4 \text{ FLOPs/Byte}}$$

### E. The Bottleneck Proof
In both cases ($51.5$ and $62.4$ FLOPs/Byte), the arithmetic intensity is **significantly lower than the GPU's hardware threshold of $187.5$ FLOPs/Byte**. 

Even during prefill, which is normally compute-bound for projection layers, **the self-attention layer is severely memory-bandwidth bound**. The GPU's Tensor Cores spend more than **$66\%$ of their time completely idle**, stalled waiting for the $[T, T]$ matrix reads and writes to finish over the VRAM memory bus.

---

## 3. FlashAttention Arithmetic Intensity Optimization

FlashAttention alters this equation by keeping all intermediate $[T, T]$ matrices in local SRAM registers. By fusing the operations into a single CUDA kernel and tiling the inputs, FlashAttention completely eliminates the quadratic $O(T^2)$ memory reads and writes.

### A. FlashAttention Memory Transfer (Bytes Read and Written to HBM)
In FlashAttention, the only memory transactions with HBM are:
1.  **Read** $Q$ of size $[T, D]$: $2 \cdot T \cdot D$ bytes.
2.  **Read** $K$ of size $[T, D]$: $2 \cdot T \cdot D$ bytes.
3.  **Read** $V$ of size $[T, D]$: $2 \cdot T \cdot D$ bytes.
4.  **Write** Output $O$ of size $[T, D]$: $2 \cdot T \cdot D$ bytes.

$$\text{Total Flash HBM Bytes} = 8 \cdot T \cdot D$$

### B. FlashAttention Arithmetic Intensity
$$\text{Arithmetic Intensity}_{\text{Flash}} = \frac{T^2 (4D + 3)}{8 \cdot T \cdot D} = \frac{T(4D + 3)}{8D}$$

For $D = 128$:
$$\text{Arithmetic Intensity}_{\text{Flash}} \approx \frac{515 \cdot T}{1024} \approx \mathbf{0.5 \cdot T}$$

### C. FlashAttention Scaling Impact
Let's see how FlashAttention scales with context length:
*   At **$T = 512$**: $\text{Arithmetic Intensity} \approx \mathbf{256 \text{ FLOPs/Byte}}$ (Exceeds the $187.5$ limit! Compute-bound, runs at peak hardware speed).
*   At **$T = 4096$**: $\text{Arithmetic Intensity} \approx \mathbf{2048 \text{ FLOPs/Byte}}$ (Massively compute-bound. Maximum Tensor Core utilization).

```
         Arithmetic Intensity vs. Context Length (D=128)
   2500 +------------------------------------------------------------+
        |                                              * Flash (2048)|
   2000 |                                                            |
        |                                                            |
   1500 |                                                            |
        |                                                            |
   1000 |                                                            |
        |                                                            |
    500 |                                                            |
        |                * Flash (256)                               |
      0 +--o--o----------o-----------------------------o------------+
        512              1024                          4096
                 --- o --- Standard (Flat 50-62 FLOPs/B)
```

By removing the VRAM memory transfer bottleneck, FlashAttention shifts attention from a slow memory-bandwidth bound operation to a highly parallel compute-bound operation, scaling linearly with available hardware FLOPs.

---

## 4. Hardware Execution Analysis (Standard vs. Flash)

The systems-level changes of FlashAttention are highlighted in this comparison:

| Metric / Feature | Standard Attention | FlashAttention |
| :--- | :--- | :--- |
| **HBM Read/Write Scaling** | $O(T^2 + TD)$ | $O(TD)$ (Linear with respect to context) |
| **SRAM Caching** | Not used for intermediate matrices | Holds $Q, K, V$ blocks of size $B_r, B_c \le 64$ KB |
| **Intermediate Storage** | $[T, T]$ matrices written to VRAM | Virtualized; only final $[T, D]$ output is written |
| **Softmax Scaling** | Global reduction (requires HBM load) | Online tracking (local registers) |
| **Prefill Spike** | Quadratic $O(T^2)$ growth | Constrained to local tiles (constant 20 MB) |

### Why VRAM Spike Shrinks to a Constant 20 MB
In our sweeps, enabling FlashAttention reduced the prefill VRAM spike from a quadratically-scaling maximum of $138$ MB at $4096$ context to a **flat $20.00$ MB** across all contexts.
1.  **Standard Attention**: Allocates space in VRAM for the full $T \times T$ softmax probabilities matrix. As $T$ scales to $4096$, the matrix sizes balloon, causing memory allocation peaks.
2.  **FlashAttention**: Computes the dot-products block-by-block using static SRAM allocations (determined by block size $B_r \times B_c$, which are fixed numbers like $64 \times 64$). Regardless of how long the sequence $T$ grows, the physical SRAM tile footprint remains identical. The only VRAM needed is for the input and output matrices, which grow slowly and linearly.

---

## Related Documentation
*   [Theory Reference: Fused Attention Kernels and SRAM Tiling](../docs/flash-attention.md)
*   [Theory Reference: KV Cache Sizing and Memory Layouts](../docs/kv-cache.md)
*   [Benchmark Report: FlashAttention Memory and Speed Sweeps](../benchmarks/02-07-2026-flash-attention-benchmark.md)
