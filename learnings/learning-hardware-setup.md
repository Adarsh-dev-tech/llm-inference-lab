# Learning Report: Hardware Setup, Compilation, and Layer Offloading

## Overview
This report details the compilation and optimization of the `llama.cpp` inference engine on an NVIDIA RTX 3050 Laptop GPU, along with the performance dynamics observed during layer offloading.

---

## 1. CUDA Compilation and Setup

Compiling `llama.cpp` from source with CUDA support on Windows was completed with the following configuration:
*   **Compilation Command/Flags**: Enabled CUDA compilation by setting the flag `-DGGML_CUDA=ON` during the CMake build generation step:
    ```bash
    cmake -B build -S . -DGGML_CUDA=ON
    cmake --build build --config Release
    ```
*   **CUDA Toolkit**: Utilizing CUDA Toolkit v13.3.
*   **Observations**: Source compilation ensures that standard instructions (such as AVX2 for the CPU and optimal CUDA compute kernels for the GPU SMs) are properly optimized for the host machine. The built binaries (`llama-cli`) natively bind to CUDA DLLs for execution.

---

## 2. Layer Offloading (`-ngl` / `--n-gpu-layers`) Sweeps

The baseline model used was `Qwen2.5-7B-Instruct` in `Q4_K_M` quantization format (32 layers). Sweeping the offloaded layers flag (`-ngl` / `--n-gpu-layers`) from 0 to 33 revealed clear performance transitions:

| Metric | CPU-Only (`-ngl 0`) | Partial GPU Offload (`-ngl 20`) | Full GPU Offload (`-ngl 33`) |
| :--- | :--- | :--- | :--- |
| **Prompt Throughput** | 16.15 tokens/sec | 35.90 tokens/sec | 91.50 tokens/sec |
| **Generation Throughput** | 9.33 tokens/sec | 21.10 tokens/sec | 35.70 tokens/sec |
| **System RAM Usage** | ~13.82 GB | ~14.56 GB | ~14.68 GB |
| **VRAM Usage** | ~1.06 GB (OS + background) | ~4.16 GB | ~4.54 GB |

### Performance Trends:
1.  **CPU-Only (`-ngl 0`)**: Yields very slow generation (9.33 t/s) and prompt processing (16.15 t/s). The workload is entirely bound by the CPU instruction pipeline and system memory bandwidth.
2.  **Partial GPU Offload (`-ngl 20`)**: Shifting 20 out of 32 transformer layers to the GPU increases generation speed to 21.10 t/s and prompt speed to 35.90 t/s. VRAM rises to ~4.16 GB.
3.  **Full GPU Offload (`-ngl 33`)**: Offloading all layers (including the final classification/output layer) achieves the highest throughput (35.70 t/s decode, 91.50 t/s prefill) while consuming a modest ~4.54 GB of VRAM. This leaves ~1.46 GB of VRAM headroom for KV cache growth within the 6 GB physical limit.

---

## 3. Systems Analysis: Memory Bandwidth vs. PCIe Bus

The dramatic performance scaling observed is governed by two hardware-level bottlenecks:

### Memory Bandwidth Bottleneck
Large Language Model decoding is a memory-bandwidth-bound operation. For each generated token, the model weights must be loaded from memory to the processor's registers. 
*   **System Host RAM**: Single-channel DDR5 RAM operates at a peak bandwidth of approximately **38.4 GB/s**.
*   **GPU VRAM**: The RTX 3050 Laptop's GDDR6 VRAM operates at a peak bandwidth of approximately **168 to 192 GB/s** (roughly 4x-5x faster).
*   **Result**: Executing layers on the GPU enables streaming the model weights much faster, directly leading to a 3.8x increase in decode speed (9.33 t/s to 35.70 t/s) and a 5.6x increase in prompt processing (16.15 t/s to 91.50 t/s).

### PCIe Bus Bottleneck (Split-Offloading)
During partial GPU offloading (`-ngl 20`), the model layers are split between host RAM and GPU VRAM.
*   **Mechanism**: During the forward pass of the transformer block, intermediate tensor activations must step across the PCIe bus back and forth between host RAM (CPU calculation) and device VRAM (GPU calculation).
*   **Bottleneck**: Even on a PCIe Gen4 x8 interface, the transfer rate (max ~16 GB/s) is significantly slower than the GPU's internal memory bus. This constant synchronization stalls execution units.
*   **Recommendation**: To achieve maximum performance, always aim to offload 100% of the model layers (`-ngl 33` for Qwen 7B) to the GPU. Only use partial offloading if the model footprint is too large to fit in VRAM.
