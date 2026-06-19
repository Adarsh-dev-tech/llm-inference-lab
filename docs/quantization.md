# Theory Reference: Quantization Precision Formats and GGUF

Quantization is the process of mapping continuous (floating-point) values to a smaller set of discrete (integer) values. In deep learning inference, quantization is the primary method used to compress Large Language Models (LLMs) to fit within resource-constrained consumer hardware.

---

## 1. Floating-Point Baselines vs. Quantized Formats

Before quantization, neural network parameters (weights) and activations are stored in high-precision floating-point formats:

*   **FP32 (Single-Precision Floating-Point)**: Uses 32 bits (1 sign, 8 exponent, 23 fraction). It serves as the baseline for model training but is highly inefficient for local inference.
*   **FP16 (Half-Precision)**: Uses 16 bits (1 sign, 5 exponent, 10 fraction). It halves the memory footprint relative to FP32 while maintaining near-identical accuracy.
*   **BF16 (Brain Floating-Point)**: Uses 16 bits (1 sign, 8 exponent, 7 fraction). It matches FP32's dynamic range but has reduced precision, preventing underflow/overflow during training.
*   **INT8 (8-bit Integer)**: Maps floating-point values to 256 discrete integer steps, reducing the memory footprint by 50% relative to FP16.
*   **INT4 (4-bit Integer)**: Maps values to 16 discrete steps, providing another 50% reduction in weight memory, enabling large models to fit on consumer GPUs.

---

## 2. Block-wise Quantization Mechanics

A naive global quantization strategy maps the minimum and maximum values of an entire weight tensor to the integer range. However, outliers (extremely large or small activations) stretch the scale factor, causing massive precision loss for the remaining weights.

To prevent this, LLM runtimes utilize **Block-wise (or Group-wise) Quantization**:
*   The weight tensor is divided into small, non-overlapping blocks of size $B$ (typically $B=32$ or $B=256$ elements).
*   For each block, a local scaling factor ($d$) and optionally a local offset (or minimum value, $m$) are calculated.
*   The weights within that block are then quantized to low-bit integers (e.g., 4-bit) relative to the block's scale:
    $$q = \text{round}\left( \frac{w}{d} \right)$$
*   The block scale ($d$) is stored alongside the quantized integer values. During inference, the weights are dequantized on-the-fly in the GPU cache/SRAM using the local scale:
    $$w \approx q \times d$$

By keeping blocks small ($B=32$), the impact of local weight outliers is isolated, preserving high reasoning accuracy.

---

## 3. The GGUF Specification

**GGUF (GGML Unified Format)** is a binary file format designed for fast loading and running of LLM models, popularized by `llama.cpp`.

Key characteristics of the GGUF specification include:
*   **Single-File Packaging**: Models are distributed as a single file containing all model metadata, configuration keys, and tensor weights.
*   **Extensible Key-Value Metadata**: The file header uses a key-value dictionary structure. This allows adding new metadata (such as custom tokenizer configurations or hyperparameter values) without breaking backward compatibility for older parsers.
*   **Memory-Mapped File Loading (`mmap`)**: GGUF structures tensors to align with memory boundaries. The file can be mapped directly into system memory (`mmap`). This allows the system to load tensors on-demand, reducing initial startup time and memory footprint because weights do not need to be parsed and copied multiple times in RAM.
*   **Unified CPU/GPU Execution**: GGUF tensors are structured to allow the runtime to offload a portion of the tensor arrays directly to GPU VRAM while streaming the remaining tensors through host CPU registers.

---

## 4. GGUF Quantization Types: `Q4_K_M`, `Q5_K_M`, and `Q8_0`

`llama.cpp` implements several specialized block-wise quantization formats (prefixed with `Q` followed by bit-width and quantization style):

### Q8_0
*   **Bit-width**: 8 bits per weight.
*   **Block size**: 32 elements.
*   **Scale**: 16-bit float (`fp16`) scale factor per block.
*   **Use Case**: Very low quality loss (near FP16 perplexity), but provides only a 2x reduction in size.

### Q4_K_M
*   **Quantization Scheme**: Standard "K-Quant" format.
*   **Structure**: Uses a mixed-precision structure. The model weights are quantized using 4-bit values for some layers (like feed-forward network blocks), but critical layers (like self-attention projection weights) are kept at a higher precision (often 5-bit or 6-bit) to preserve reasoning capability.
*   **Block size**: 256 elements.
*   **Use Case**: The standard industry default for consumer GPUs. It balances high generation throughput with minimal reasoning quality degradation.

### Q5_K_M
*   **Quantization Scheme**: Similar mixed-precision block structure to `Q4_K_M` but allocates 5 bits per weight for most layers.
*   **Use Case**: Acts as a middle ground between `Q4_K_M` and `Q8_0`. It recovers a significant portion of the quantization perplexity loss of `Q4_K_M` while requiring less VRAM than `Q8_0`.
