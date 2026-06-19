# LLM Inference Benchmarking Methodology

This reference document establishes the experimental controls, statistical rigor, and methodology for conducting LLM inference benchmarks on local consumer-grade hardware.

---

## 1. Local Hardware Constraints & Bottlenecks

Benchmarking on consumer laptops (specifically targets like an NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM and a 12th Gen Intel Core i5 CPU) requires understanding constraints that do not exist in cloud datacenter environments:

*   **VRAM Capacity Limitations**: A 6 GB VRAM budget restricts the footprint of model weights and active KV caches. Exceeding this boundary triggers dynamic offloading to host system RAM, causing a performance cliff (drop in TPS by 10x-20x), or a CUDA Out-of-Memory (OOM) crash.
*   **Thermal Throttling**: Laptops operate in highly restricted thermal envelopes. Sustained GPU and CPU workloads raise chip temperatures rapidly, triggering hardware clocks to down-throttle to prevent damage. This introduces latency drift, where later runs are slower than initial runs.
*   **Power Limit Envelopes**: Mobile GPUs operate within dynamic Total Graphics Power (TGP) budgets (typically 35W–80W). Dynamic power sharing (e.g., NVIDIA Dynamic Boost) shifts power allocation between the CPU and GPU depending on the load, introducing variance.
*   **System Bus & PCIe Bottlenecks**: Partial layer offloading requires copying tensor activations between host RAM (CPU) and GPU VRAM over the PCIe bus (often restricted to PCIe Gen4 x8 or Gen3 x4 on mobile boards). This bus becomes a major bottleneck for inter-layer data transfer.

---

## 2. Experimental Controls for Hardware Stability

To eliminate hardware-induced noise and guarantee reproducibility, the following protocols must be strictly followed before and during benchmark execution:

### Control 1: Thermal Stabilization (Cool-down Clocks)
To prevent thermal throttling from skewing execution times, the system must cool down between runs:
*   Allow the hardware to idle for a minimum of **60 seconds** between benchmarks.
*   Log GPU core temperatures dynamically using NVML (or `nvidia-smi`) to ensure starting temperatures are within a consistent baseline range ($\pm 2^{\circ}\text{C}$).

### Control 2: Power and Performance Mode
*   Keep the laptop connected to **AC power** at all times. Battery discharge rates cannot sustain the wattage needed for full GPU boost states.
*   Configure the Operating System power profile to **"Best Performance"**.
*   Disable dynamic clock scaling or frame-rate limiters (e.g., NVIDIA Battery Boost or WhisperMode).

### Control 3: System Environment Isolation
*   Terminate all non-essential user space applications (browsers, IDEs, chat clients, discord) to maximize free RAM and VRAM.
*   Temporarily suspend background system tasks (such as Windows Defender active file scanning, Windows Update, or OneDrive synchronization) to minimize CPU context switching and OS scheduler jitter.

---

## 3. Runtime Controls & Warm-up

Software runtimes (like `llama.cpp` or PyTorch) dynamically compile execution graphs, cache memory addresses, and initialize CUDA context parameters during their first forward pass.

*   **Warm-up Iterations**: Always conduct **3 to 5 warm-up runs** (generating a small number of tokens) before recording measurements. Discard these warm-up runs from the statistical analysis. Failing to do so will artificially inflate the measured Time-to-First-Token (TTFT) and average latency.
*   **State Reset**: Reset the execution context's memory allocation and internal C++ performance counters (e.g., calling `llama_perf_context_reset`) before starting the measured loops.

---

## 4. Statistical Rigor and Metrics Selection

A single benchmark execution run is insufficient due to background OS scheduler noise. All benchmark metrics must be reported with statistical distributions across $N \ge 30$ iterations:

### Core Latency Metrics
1.  **Time-to-First-Token (TTFT)**: The duration from prompt submission to the first generated token. Measures prefill execution efficiency (compute-bound).
2.  **Inter-Token Latency (ITL)**: The time elapsed between two consecutive output tokens during decode. Measures token-by-token generation consistency.
3.  **End-to-End Latency**: The total request duration.

### Statistical Summaries
For each metric, compute and report:
*   **Mean ($\bar{x}$)**: General average performance.
*   **Median (p50)**: Typical user experience, robust to outlier runs.
*   **95th and 99th Percentiles (p95, p99)**: Measures tail latency spikes, highlighting system stutter or scheduling hiccups.
*   **Standard Deviation ($s$)**: Indicates stability and variation.
*   **95% Confidence Interval (CI)**:
    $$\text{CI} = \bar{x} \pm t^* \left( \frac{s}{\sqrt{n}} \right)$$
    Where $t^*$ is the critical value from the Student's $t$-distribution and $n$ is the number of recorded runs. This interval helps verify if performance differences between configurations are statistically significant or merely statistical noise.
