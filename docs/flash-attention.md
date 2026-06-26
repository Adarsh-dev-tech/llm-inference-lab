# Theory Reference: FlashAttention Fused Kernels and SRAM Tiling

Traditional Transformer self-attention is highly inefficient on modern hardware because it is **memory-bandwidth bound**. This guide details the hardware bottlenecks of standard attention and provides a complete mathematical and systems deconstruction of **FlashAttention** (Dao et al., 2022) to serve as a comprehensive, self-contained reference.

---

## 1. The Core Bottleneck: High-Bandwidth Memory (HBM) IO

On modern GPU hardware (like the RTX 3050), computation speed (measured in Floating Point Operations Per Second, or FLOPS) is extremely fast. However, moving data from the GPU's large video memory (**High-Bandwidth Memory / HBM**) to the GPU's local processors is relatively slow. 

### GPU Memory Hierarchy
To understand the bottleneck, we must look at the latency and capacity layers of a GPU:

```
+------------------------------------+--------------------------+------------------------+
| Memory Layer                       | Typical Capacity         | Latency / Bandwidth    |
+------------------------------------+--------------------------+------------------------+
| HBM (VRAM / Global Memory)         | 6 GB - 80 GB             | ~1.5 - 2.0 TB/s (Slow) |
| L2 Cache (On-Chip Shared)          | 32 MB - 96 MB            | ~3.0 - 5.0 TB/s        |
| SRAM (Shared Memory / Registers)   | ~100 KB per Streaming MP | ~19 TB/s (Ultra-Fast)  |
+------------------------------------+--------------------------+------------------------+
```

### HBM Bottleneck in Standard Attention
In standard self-attention, the formula is:
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

Given a sequence length $T$ and head dimension $D$, the intermediate attention matrix $S = \frac{QK^T}{\sqrt{d_k}}$ and the softmax probabilities matrix $P = \text{softmax}(S)$ are of shape **$[T, T]$**. 
For $T = 4096$ tokens, a single matrix of shape $[4096, 4096]$ at FP16 contains **$16.7$ Million elements** ($\approx 33.5$ MB).

The hardware execution steps for standard attention require repeatedly reading and writing these intermediate matrices back to HBM:

```
  HBM (VRAM)                                            GPU SRAM (Registers)
+--------------+                                           +------------+
| Q, K vectors | ======= (1) Read Q and K ===============> |  Multiply  |
+--------------+                                           +------------+
                                                                 |
                                                                 v
+--------------+                                           +------------+
|  S Matrix    | <====== (2) Write S [T x T] to HBM ------ |   Store    |
+--------------+                                           +------------+
                                                                 |
                                                                 v
+--------------+                                           +------------+
|  S Matrix    | ======= (3) Read S to compute Softmax ==> |   Softmax  |
+--------------+                                           +------------+
                                                                 |
                                                                 v
+--------------+                                           +------------+
|  P Matrix    | <====== (4) Write P [T x T] to HBM ------ |   Store    |
+--------------+                                           +------------+
                                                                 |
                                                                 v
+--------------+                                           +------------+
|  P, V vectors| ======= (5) Read P and V ===============> |  Multiply  |
+--------------+                                           +------------+
                                                                 |
                                                                 v
+--------------+                                           +------------+
|  O Matrix    | <====== (6) Write O [T x D] to HBM ------ |   Output   |
+--------------+                                           +------------+
```

*   **The Inefficiency**: The GPU's processor cores are forced to sit idle, waiting for the paged $T \times T$ intermediate matrices to be read and written over the HBM bandwidth interface. The memory transfer scaling is **quadratic ($O(T^2)$)**.

---

## 2. The FlashAttention Solution: Tiling and Kernel Fusion

FlashAttention resolves this bottleneck by restructuring the self-attention computation so that the intermediate $[T, T]$ matrices are **never written to HBM**. It does this using two concepts:

1.  **Kernel Fusion**: Combining the projection, scaling, masking, softmax, and value multiplication into a single GPU CUDA kernel.
2.  **SRAM Tiling**: Slicing the input Query, Key, and Value matrices into small blocks (tiles) that fit completely inside the GPU's ultra-fast **SRAM (Shared Memory)**.

### Tiling Execution Steps
1.  Divide the Query matrix $Q$ into blocks of size $B_c \times D$ and the Key ($K$) and Value ($V$) matrices into blocks of size $B_r \times D$.
2.  Load a block of $Q$ and a block of $K$ and $V$ into SRAM.
3.  Compute the local attention dot-products for these blocks inside SRAM.
4.  Perform incremental softmax updates (see Section 3).
5.  Multiply the local softmax scores by the local $V$ block to update a running output accumulator $O$.
6.  Write only the final output block of $O$ back to HBM.

By keeping the intermediate computations entirely inside the fast SRAM register file, FlashAttention reduces the HBM memory access complexity from **quadratic ($O(T^2)$)** to **linear ($O(T)$)**.

---

## 3. The Mathematics of Online Softmax Tracking

Softmax is defined as:
$$\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad \text{where } m = \max_j x_j$$

### The Tiling Challenge
Standard softmax is non-local: to calculate the denominator sum $\sum e^{x_j - m}$ for any element in a row, you must already know the **global maximum ($m$)** of the entire row. 
When tiling, we load blocks of a row sequentially. We do not have access to the future elements of the row. If we compute softmax locally for block 1, and then discover a larger value in block 2, our previous softmax denominators and maximums become mathematically invalid.

### The Online Softmax Trick
FlashAttention resolves this by tracking and updating three running values block-by-block for each row of the attention matrix:
1.  **$m^{(i)}$**: The running maximum value of the row up to block $i$.
2.  **$d^{(i)}$**: The running sum of exponentials (denominator) up to block $i$.
3.  **$O^{(i)}$**: The running output vector up to block $i$.

#### Step-by-Step Update Equations
Suppose we have computed the running state up to block $i-1$, yielding maximum $m^{(i-1)}$, denominator $d^{(i-1)}$, and output $O^{(i-1)}$. 

When we load block $i$ (with local maximum $\tilde{m}$ and local exponent sum $\tilde{d}$):

1.  **Update the Running Maximum**:
    $$m^{(i)} = \max\left(m^{(i-1)}, \tilde{m}\right)$$

2.  **Rescale and Merge the Denominator**:
    To combine the previous sum $d^{(i-1)}$ and the new sum $\tilde{d}$, we must rescale both to align with the new maximum $m^{(i)}$:
    $$d^{(i)} = d^{(i-1)} e^{m^{(i-1)} - m^{(i)}} + \tilde{d} e^{\tilde{m} - m^{(i)}}$$

3.  **Rescale the Running Output Accumulator**:
    We adjust the previous output $O^{(i-1)}$ by the rescaling factor and add the new local block output contribution $\tilde{O}$:
    $$O^{(i)} = O^{(i-1)} \frac{d^{(i-1)} e^{m^{(i-1)} - m^{(i)}}}{d^{(i)}} + \tilde{O} \frac{\tilde{d} e^{\tilde{m} - m^{(i)}}}{d^{(i)}}$$

Once all blocks are processed, the final accumulator $O^{(N)}$ is mathematically identical to the standard global softmax output, calculated without ever storing the intermediate $[T, T]$ matrix in HBM.

---

## 4. Concrete Example of Online Softmax

Let's trace a simplified vector of raw attention dot-products divided into two tiles:
$$\text{Vector } X = [2.0, 4.0, 1.0, 5.0]$$
Divided into two tiles of size 2:
$$\text{Tile 1 } (i=1): X_1 = [2.0, 4.0], \quad \text{Tile 2 } (i=2): X_2 = [1.0, 5.0]$$

### Execution Trace:

#### **Pass 1: Process Tile 1** ($X_1 = [2.0, 4.0]$)
1.  Local maximum: $\tilde{m}_1 = \max(2.0, 4.0) = \mathbf{4.0}$
2.  Local sum of exponentials: $\tilde{d}_1 = e^{2.0 - 4.0} + e^{4.0 - 4.0} = e^{-2} + e^0 \approx 0.135 + 1.0 = \mathbf{1.135}$
3.  Initialize running states:
    *   $m^{(1)} = \tilde{m}_1 = \mathbf{4.0}$
    *   $d^{(1)} = \tilde{d}_1 = \mathbf{1.135}$
4.  Compute initial output contribution (assuming Value vector components are 1 for simplicity):
    *   $O^{(1)} = [e^{2-4}, e^{4-4}] / 1.135 = [0.119, 0.881]$

#### **Pass 2: Process Tile 2** ($X_2 = [1.0, 5.0]$)
1.  Local maximum: $\tilde{m}_2 = \max(1.0, 5.0) = \mathbf{5.0}$
2.  Local sum of exponentials: $\tilde{d}_2 = e^{1.0 - 5.0} + e^{5.0 - 5.0} = e^{-4} + e^0 \approx 0.018 + 1.0 = \mathbf{1.018}$
3.  Update running maximum:
    $$m^{(2)} = \max\left(m^{(1)}, \tilde{m}_2\right) = \max(4.0, 5.0) = \mathbf{5.0}$$
4.  Update running denominator:
    $$d^{(2)} = d^{(1)} e^{m^{(1)} - m^{(2)}} + \tilde{d}_2 e^{\tilde{m}_2 - m^{(2)}}$$
    $$d^{(2)} = (1.135) e^{4.0 - 5.0} + (1.018) e^{5.0 - 5.0}$$
    $$d^{(2)} = (1.135 \times 0.3678) + (1.018 \times 1.0) \approx 0.417 + 1.018 = \mathbf{1.435}$$
5.  Update running output accumulator (rescaled to the new global maximum):
    *   $O^{(2)}$ incorporates the new denominator $1.435$ and matches the exact global softmax:
        $$\text{Global Softmax} = [e^{2-5}, e^{4-5}, e^{1-5}, e^{5-5}] / d^{(2)}$$
        $$\text{Global Softmax} = [0.0498, 0.3678, 0.0183, 1.0] / 1.435 = [0.035, 0.256, 0.013, 0.696]$$

---

## 5. FlashAttention-2 Improvements

FlashAttention-2 (Dao, 2023) further optimizes execution speed by addressing warp occupancy and thread-level scaling:
1.  **Fewer Non-MatMul Operations**: Rearranges the scaling math to perform fewer divisions and exponent operations, which are slow on GPU CUDA cores compared to fused multiply-add (FMA) matrix operations.
2.  **Better Work Partitioning**: Parallelizes the attention calculation across the sequence length dimension ($T$) as well as batch and head dimensions. This keeps more GPU Streaming Multiprocessors (SMs) saturated on smaller batches or long single sequences.
3.  **Warp Specialization**: Structures memory loads and matrix multiplications dynamically to prevent thread synchronization stalls inside warp groups.

---

## Related Documentation
*   [Theory Reference: Transformer Architectures and Self-Attention Mechanics](transformer-basics.md)
*   [Theory Reference: KV Cache Sizing and Memory Layouts](kv-cache.md)
*   [Benchmark Report: FlashAttention Memory and Speed sweeps](../benchmarks/25-06-2026-flash-attention-benchmark.md)
