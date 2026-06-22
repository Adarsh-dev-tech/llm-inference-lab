# Learning Report: Transformer Weight Distribution, KV Caching, and Context Limits

## Overview
This report maps the systems-level implementation details of Transformer internals, detailing parameter distribution, the mathematical savings of Key-Value (KV) caching, and how context scaling boundaries emerge on a 6 GB VRAM consumer GPU.

---

## 1. Transformer Weight Distribution Map

In a decoder-only model like `Qwen2.5-7B`, parameters are distributed between the Self-Attention block, the Feed-Forward Network (FFN/MLP) block, and the vocabulary embedding layers. 

### Parameter Breakdown Analysis
*   **Model Configuration (Qwen2.5-7B)**:
    *   Hidden dimension ($C$) = 3584
    *   Number of layers ($L$) = 28
    *   Number of query heads ($H_q$) = 28, head dimension ($D$) = 128
    *   Number of key-value heads ($H_{kv}$) = 4 (Grouped-Query Attention)
    *   FFN intermediate dimension ($d_{ffn}$) = 18944
    *   Vocabulary size ($V_{\text{vocab}}$) = 151936

### Parameter Allocations:
1.  **Embedding Layers (Token & Position)**:
    *   Formula: $C \times V_{\text{vocab}}$
    *   Parameters: $3584 \times 151936 \approx 544.5$ Million (~7.1% of total model size).
2.  **Self-Attention Block (Per Layer)**:
    *   Projections: $Q, K, V$ projection weights + Output projection ($W^O$).
    *   With Grouped-Query Attention ($H_{kv} = 4$), the Key and Value matrices are smaller:
        *   $W^Q$: $C \times C = 3584 \times 3584 \approx 12.84$ Million.
        *   $W^K, W^V$: $C \times (H_{kv} \times D) = 3584 \times (4 \times 128) \approx 1.84$ Million each.
        *   $W^O$: $C \times C = 3584 \times 3584 \approx 12.84$ Million.
    *   Total Attention parameters per layer: $\approx 29.36$ Million.
3.  **Feed-Forward Network Block (Per Layer - SwiGLU)**:
    *   Projections: Gate weight ($W^{\text{gate}}$), Up weight ($W^{\text{up}}$), Down weight ($W^{\text{down}}$).
    *   Formulas:
        *   $W^{\text{gate}}, W^{\text{up}}$: $C \times d_{ffn} = 3584 \times 18944 \approx 67.89$ Million each.
        *   $W^{\text{down}}$: $d_{ffn} \times C = 18944 \times 3584 \approx 67.89$ Million.
    *   Total FFN parameters per layer: $\approx 203.67$ Million.

### Core Insight:
The Feed-Forward Network (MLP) block contains **~6.9 times more parameters** than the Self-Attention block per layer (203.67M vs 29.36M). Consequently, FFN layers account for **over 75%** of the static weight-streaming bandwidth required during a forward pass. 

---

## 2. Dynamic Caching Overhead: KV Cache vs. Recomputation

Autoregressive text generation requires predicting the next token $t+1$ based on all preceding tokens $1$ to $t$. 

### Without Key-Value (KV) Caching (Recomputation)
At each decoding step $t$:
1.  All tokens $1$ to $t$ must be passed through the projection layers: $Q = X W^Q, K = X W^K, V = X W^V$.
2.  This requires $O(t)$ matrix multiplications. 
3.  The self-attention dot product $Q K^T / \sqrt{d_k}$ scales quadratically ($O(t^2)$).
4.  **Bottleneck**: Computing projections for all historical tokens at every single step introduces massive computational overhead, stalling generation speed as sequence length grows.

### With Key-Value (KV) Caching
We exploit the causal property of the decoder: the keys and values of past tokens do not change.
1.  **Store past states**: Save the Key ($K_{1 \dots t-1}$) and Value ($V_{1 \dots t-1}$) matrices in VRAM.
2.  **Single projection**: At step $t$, project only the single new input token $t$ to obtain its Query ($q_t$), Key ($k_t$), and Value ($v_t$) vectors ($O(1)$ projections).
3.  **Concatenate**: Append $k_t$ and $v_t$ to the cached memory matrices:
    $$K_{\le t} = [K_{\le t-1}; k_t], \quad V_{\le t} = [V_{\le t-1}; v_t]$$
4.  **Attention Map**: Compute attention scores between the single query vector $q_t$ and all keys in cache:
    $$\text{Attention}_t = \text{softmax}\left( \frac{q_t K_{\le t}^T}{\sqrt{d_k}} \right) V_{\le t}$$
5.  **Computational Savings**: Projections drop from $O(t)$ to $O(1)$. The attention dot-product scales linearly ($O(t)$ operations per step).

### Caching Overhead & Memory Bandwidth Shift
While KV caching bypasses the $O(t^2)$ compute bottleneck, it shifts the system bottleneck from **compute-bound** to **memory-bandwidth-bound**:
*   **VRAM Allocation**: The KV cache must be kept in fast GPU memory. As context length increases, the cache grows and can exhaust VRAM.
*   **Memory Transfer**: At each step, the entire historical KV cache must be read from VRAM into the GPU's SRAM registers to calculate the attention map. This stresses GPU memory bandwidth rather than compute cores.

---

## 3. Context Limit Mathematics on a 6 GB VRAM Budget

To model when our local RTX 3050 Laptop GPU will hit the out-of-memory (OOM) cliff, we analyze the scaling limits of Qwen2.5-7B's memory requirements.

### Caching Formulas:
$$\text{KV Cache Size (bytes)} = 2 \times \text{layers} \times H_{kv} \times D \times \text{bytes\_per\_param} \times \text{context\_len} \times \text{batch\_size}$$

For `Qwen2.5-7B` at FP16 (2 bytes per parameter) with Grouped-Query Attention ($H_{kv} = 4$):
$$\text{Size per Token} = 2 \times 28 \times 4 \times 128 \times 2 = 57,344 \text{ bytes} \approx 57.34 \text{ KB/token}$$

### Context Length scaling footprint (Batch Size = 1):
*   **1,024 tokens**: $1024 \times 57.34 \text{ KB} \approx 58.7 \text{ MB}$
*   **4,096 tokens**: $4096 \times 57.34 \text{ KB} \approx 234.9 \text{ MB}$
*   **16,384 tokens**: $16384 \times 57.34 \text{ KB} \approx 939.5 \text{ MB}$
*   **32,768 tokens**: $32768 \times 57.34 \text{ KB} \approx 1.88 \text{ GB}$

### The Power of Grouped-Query Attention (GQA)
If the model used standard Multi-Head Attention (MHA) where $H_{kv} = H_q = 28$:
$$\text{MHA Size per Token} = 2 \times 28 \times 28 \times 128 \times 2 = 401,408 \text{ bytes} \approx 401.4 \text{ KB/token}$$
*   At **4,096 tokens**, the cache alone would require **~1.64 GB** of VRAM (a **7x increase** compared to GQA's 234.9 MB).
*   GQA makes long-context local inference possible by compressing Key-Value heads, leaving critical VRAM capacity for the model weights.

### Maximum Context Limits on the RTX 3050 (6 GB VRAM)
Under `Q4_K_M` quantization (weights footprint $\approx 4.92$ GB, leaving $\approx 1.08$ GB VRAM headroom):
*   **Max Theoretical Context**: $\approx 1.08 \text{ GB} / 57.34 \text{ KB/token} \approx 19,000$ tokens.
*   **Real-world Limit**: Taking into account activation memory spikes during the prefill phase (~0.30 GB) and OS baseline memory drift, the practical limit is **~12,000 to 14,000 tokens** before triggering dynamic PCIe memory paging and collapsing generation speed.

---

## Related Theory Documents
*   [Theory Reference: Transformer Architectures and Self-Attention Mechanics](../docs/transformer-basics.md)

