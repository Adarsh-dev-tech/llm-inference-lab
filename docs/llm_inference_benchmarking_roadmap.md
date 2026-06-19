# LLM Inference Benchmarking & Measurement Roadmap

This handbook serves as a comprehensive systems-engineering guide for measuring, profiling, and optimizing Large Language Model (LLM) inference systems. It is structured to build a rigorous foundation in metrics definition, benchmarking methodology, and systematic hardware-software optimization.

---

## Part 0: Local Hardware Benchmarking Strategy

### Consumer Hardware Constraints
*   **Limited VRAM**: With only 6 GB of VRAM, running modern models (like Llama-3 8B) at native FP16 precision (requiring ~16 GB) is physically impossible. Even quantized configurations (like INT4 GGUF models, requiring ~4.5 GB) leave very little remaining VRAM for the KV cache allocation. Exceeding this boundary triggers immediate out-of-memory (OOM) errors or dynamic offloading to slow system RAM, causing a performance collapse.
*   **CPU Offloading**: The `llama.cpp` runtime allows execution layers to be split between CPU and GPU. While this enables running larger parameters (e.g., Q8_0 quantizations or 13B models) on low-VRAM systems, executing transformer layers on the CPU requires transferring intermediate activations back and forth across the system bus, which severely degrades tokens-per-second.
*   **Thermal Throttling**: Laptop cooling solutions are space-constrained. Sustained benchmarking loops generate high thermal loads, causing the GPU and CPU to drop their operational clocks to prevent thermal runaway. This results in latency drift between early and late runs of a benchmark.
*   **Laptop Power Limits**: Mobile GPUs have strict Total Graphics Power (TGP) envelopes (often 35W to 80W for the RTX 3050 Laptop, compared to 450W for a desktop RTX 4090). Systems often dynamically distribute power between the CPU and GPU (e.g., NVIDIA Dynamic Boost), which introduces runtime variability.
*   **PCIe Bottlenecks**: Laptop motherboards often restrict PCIe links (e.g., PCIe Gen4 x8 or Gen3 x4/x8 instead of Gen4/Gen5 x16). When split-offloading, transferring tensors between host RAM and GPU VRAM becomes heavily bottlenecked by the interface's low transfer rate.
*   **RAM Limitations**: 16 GB of DDR5 host RAM must accommodate the operating system, IDE, background processes, and the CPU-bound portion of the model. DDR5 RAM bandwidth (~38 to 48 GB/s) is a major performance bottleneck compared to GPU memory bandwidth (~168 to 192 GB/s on GDDR6 RTX 3050), slowing down CPU calculations.

### Why Consumer Hardware Matters
*   **Edge Deployment**: Deploying LLMs on personal consumer machines or embedded systems is the primary mechanism for offline-capable, private applications.
*   **Local AI**: Removing reliance on commercial cloud APIs ensures zero operational usage fees, complete data confidentiality, and system independence.
*   **Cost-Efficient Inference**: Designing systems to work within consumer hardware boundaries democratizes AI research, allowing developers to prototype systems without renting enterprise clusters.
*   **Resource-Constrained Optimization**: Developing methods under VRAM constraints forces researchers to build a deep understanding of memory efficiency, leading to code that is highly optimized when deployed on enterprise hardware.

### Benchmarking Rules for Consumer Hardware
1.  **Close All Background Applications**: Terminate web browsers, editors, and background processes (like Game Bar, Discord, or Steam) to free up maximum RAM and VRAM.
2.  **Monitor Hardware Temperatures**: Always log GPU and CPU core temperatures. Allow the laptop to idle for 60 seconds between benchmarks to cool down and prevent thermal throttling from skewing the results.
3.  **Run Plugged Into Power**: Laptop batteries cannot supply sufficient wattage for full GPU boost states. Always run benchmarks with the laptop connected to AC power and the OS power profile set to "Best Performance".
4.  **Discard Warm-Up Runs**: Always perform at least 3-5 warm-up iterations. Runtimes dynamically compile operations and cache memory addresses during the first few queries, which skew early latency measurements.
5.  **Minimize OS Scheduling Jitter**: Disable background system scans (such as Windows Defender active file scanning or system updates) to keep CPU context switching low during execution.

### Metrics That Matter Most for This Hardware
*   **Tokens Per Second (TPS)**: The final indicator of user readability.
*   **First Token Latency (TTFT)**: Crucial for evaluating how slow the prefill phase is when split-offloaded.
*   **VRAM Utilization**: Tracking exactly how much memory the model weights and KV cache consume to avoid OOM limits.
*   **Host RAM Utilization**: Ensuring the OS does not write memory pages to disk, which halts execution.
*   **GPU Layer Offloading Count**: Tracking how throughput scales as layers are shifted from CPU to GPU.
*   **Context Length Limits**: Setting limits to prevent the dynamic KV cache from overflowing the remaining VRAM.

---

## Part 1: Core Latency Metrics

Latency metrics measure the time taken to process inputs and generate outputs. For interactive LLM applications, understanding the breakdown of latency components is crucial, as different phases of generation stress different hardware resources (compute-bound vs. memory-bandwidth-bound).

### 1. First Token Latency (TTFT)
*   **Definition**: The duration between submitting a prompt to the inference engine and receiving the very first token of the output.
*   **Why it matters**: TTFT directly determines the perceived responsiveness of an interactive chat or streaming agent. High TTFT results in a sluggish user interface.
*   **How it is measured**: Start a high-precision timer when the request payload is parsed and the prefill phase starts. Stop the timer when the first token's token ID is decoded from the model's logits and ready for serialization/streaming.
*   **Units**: Milliseconds (ms) or Seconds (s).
*   **Common mistakes**: 
    1. Including network transit time (RTT) or HTTP connection handshake overhead when measuring engine performance.
    2. Failing to synchronize CUDA operations before stopping the timer on the host side.
*   **How to collect it in Python**:
    ```python
    import time
    import torch

    # Assuming a pre-initialized tokenizer and model
    input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.cuda()

    # Synchronize before starting the clock
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Prefill phase (Prompt Processing)
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)

    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - start_time) * 1000
    print(f"TTFT: {ttft_ms:.2f} ms")
    ```
*   **Expected ranges**: 
    *   *Local GPUs (e.g., RTX 4090, 8B model, fp16)*: 15 ms to 50 ms for short prompts (<512 tokens).
    *   *Cloud APIs (e.g., A100/H100)*: 100 ms to 300 ms (includes API gateway overhead).
*   **How it impacts user experience**: A TTFT under 200 ms feels instantaneous; a TTFT above 1.5 seconds leads to users assuming the system is frozen.
*   **How it impacts serving costs**: The prefill phase is compute-bound (Matrix Multiplication-heavy). Longer prompts require massive parallel processing, consuming substantial GPU compute cycles and blocking concurrent decode phases.
*   **How major inference systems optimize it**:
    *   **Chunked Prefill**: Splitting long prompts into smaller chunks and interleaving them with decode steps to prevent stalling other users.
    *   **Prompt Caching**: Saving KV cache states for common prefixes (e.g., system prompts) to bypass prefill calculations entirely.

---

### 2. End-to-End Latency
*   **Definition**: The total time taken from the initial submission of a prompt until the final token is generated and the model ceases execution (e.g., due to encountering an `[EOS]` token or reaching a maximum token limit).
*   **Why it matters**: Defines the overall duration of batch tasks like offline summarization, translation, or document extraction.
*   **How it is measured**: Measured using a monotonic system clock from the start of request processing until the termination condition of the autoregressive loop is reached.
*   **Units**: Milliseconds (ms) or Seconds (s).
*   **Common mistakes**: Confusing local execution latency with network-facing latency, or neglecting the impact of output length variability on the total duration.
*   **How to collect it in Python**:
    ```python
    import time
    import torch

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Autoregressive generation
    output_ids = model.generate(input_ids, max_new_tokens=50, use_cache=True)

    torch.cuda.synchronize()
    e2e_latency_ms = (time.perf_counter() - start_time) * 1000
    print(f"End-to-End Latency: {e2e_latency_ms / 1000:.2f} s")
    ```
*   **Expected ranges**: Dependent on the generation length. For a 100-token generation: 1.5 s to 5.0 s.
*   **How it impacts user experience**: Crucial for batch operations. In interactive applications, it matters less than TTFT and Inter-token Latency, provided streaming is active.
*   **How it impacts serving costs**: It is the primary window during which GPU resources (specifically VRAM allocation for KV caches) are tied up by a single session.
*   **How major inference systems optimize it**: Early termination using stop sequences, speculative decoding, and optimized kernel backends (TensorRT-LLM, vLLM).

---

### 3. Prompt Processing Time (Prefill Time)
*   **Definition**: The exact time required by the model to ingest the prompt tokens, compute their activations, and initialize the Key-Value (KV) cache prior to entering the autoregressive generation phase.
*   **Why it matters**: Prefill isolates the compute-bound phase of LLM inference. Tracking it independently helps identify performance bottlenecks in attention mechanisms and batch tokenization.
*   **How it is measured**: The duration of the first forward pass of the model, processing the entire input sequence in parallel.
*   **Units**: Milliseconds (ms).
*   **Common mistakes**: Counting the serialization/deserialization of tokens as part of the raw forward pass time.
*   **How to collect it in Python**:
    ```python
    torch.cuda.synchronize()
    start_prefill = time.perf_counter()

    with torch.no_grad():
        outputs = model(input_ids) # Forward pass on the full sequence

    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - start_prefill) * 1000
    ```
*   **Expected ranges**: 0.05 ms to 2 ms per prompt token depending on batch size and model dimension (e.g., A100 GPU running Llama-3-70B).
*   **How it impacts user experience**: Directly shapes TTFT. For long contexts (e.g., RAG systems containing 10k+ tokens), prefill time is the dominant factor in TTFT.
*   **How it impacts serving costs**: Higher prefill times demand massive compute resources. If the prefill phase is not highly optimized, it limits the serving throughput of long-context applications.
*   **How major inference systems optimize it**: FlashAttention-2, Flash-Decoding, and custom kernel fusions for the self-attention projection layers.

---

### 4. Generation Time (Decode Time)
*   **Definition**: The cumulative time spent generating all output tokens, excluding the initial prefill/prompt processing phase.
*   **Why it matters**: Isolates the memory-bandwidth-bound phase of LLM inference where weights are loaded from VRAM to SRAM one layer at a time for each generated token.
*   **How it is measured**: The time elapsed from the generation of the first token until model termination.
*   **Units**: Milliseconds (ms) or Seconds (s).
*   **Common mistakes**: Failing to subtract the TTFT/prefill time from the total execution duration.
*   **How to collect it in Python**:
    ```python
    # Measure total time and prefill time separately, then subtract:
    generation_time_ms = e2e_latency_ms - ttft_ms
    ```
*   **Expected ranges**: 10 ms to 40 ms per token generated (translating to 25 to 100 tokens/sec).
*   **How it impacts user experience**: Directly determines the visual flow rate of streaming text. If generation time per token is slower than human reading speed (~200-300 words per minute, or ~15-20 tokens/sec), the interface feels sluggish.
*   **How it impacts serving costs**: Because the decode phase is memory-bandwidth bound, it is highly inefficient for a batch size of 1. Low GPU utilization during decode phases represents the single largest driver of operational serving costs.
*   **How major inference systems optimize it**: KV cache management (PagedAttention), Weight Quantization (INT4/INT8) to reduce model weight footprint and transfer time, and Tensor Parallelism.

---

### 5. Inter-token Latency (ITL)
*   **Definition**: The time elapsed between two consecutive output tokens during the autoregressive decode phase.
*   **Why it matters**: Measures the consistency of the generation speed. High variance in ITL results in "stuttering" during text streaming.
*   **How it is measured**: Record timestamps for each step of the autoregressive loop, computing the difference between adjacent timestamps.
*   **Units**: Milliseconds (ms).
*   **Common mistakes**: Ignoring scheduler-induced delays in multi-tenant environments where requests are dynamically paused/resumed.
*   **How to collect it in Python**:
    ```python
    import time
    import torch

    input_ids = tokenizer("Describe quantum physics:", return_tensors="pt").input_ids.cuda()
    past_key_values = None
    next_token_id = input_ids
    itl_list = []

    with torch.no_grad():
        for i in range(10):
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            
            outputs = model(next_token_id, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            
            torch.cuda.synchronize()
            step_end = time.perf_counter()
            
            # Skip the first iteration as it includes prefill time
            if i > 0:
                itl_list.append((step_end - step_start) * 1000)
                
    print(f"Average Inter-token Latency: {sum(itl_list)/len(itl_list):.2f} ms")
    ```
*   **Expected ranges**: 8 ms to 35 ms.
*   **How it impacts user experience**: Directly determines the stability of the streaming output text. ITL should ideally remain constant (low jitter).
*   **How it impacts serving costs**: ITL increases linearly with the batch size unless optimized. Lower ITL at higher batch sizes means more compute density per watt.
*   **How major inference systems optimize it**:
    *   **Continuous Batching**: Promptly injecting new requests into the batch as soon as current requests finish, avoiding global synchronization.
    *   **TensorRT-LLM/vLLM custom decode kernels**: Bypassing PyTorch framework overheads.

---

## Part 2: Throughput Metrics

Throughput metrics define the volume of data processed by an inference engine over a given time interval. High throughput is the goal of any commercial serving infrastructure.

```
+-----------------------------------------------------------------------+
|                       LLM Inference Bottlenecks                       |
+-----------------------------------------------------------------------+
|                                                                       |
|  [Prefill Phase] (Compute-Bound)                                      |
|  * Stresses GPU Tensor Cores / Compute Units                          |
|  * High FLOPS Utilization, Low VRAM Transfer bottleneck               |
|                                                                       |
|  [Decode Phase] (Memory-Bandwidth-Bound)                              |
|  * Stresses GPU High Bandwidth Memory (HBM) Speed                     |
|  * Low FLOPS Utilization, High VRAM Transfer bottleneck               |
|                                                                       |
+-----------------------------------------------------------------------+
```

### 1. Tokens Per Second (TPS)
*   **Definition**: The total number of tokens generated by the system per second. Can be calculated per-stream or globally across the system.
*   **Why it matters**: The fundamental velocity metric of an inference engine. Used to compare engines, hardware, and quantization methods.
*   **How it is measured**: Divide the number of generated output tokens by the generation time (in seconds).
*   **Units**: Tokens per second (t/s or tokens/sec).
*   **Common mistakes**: 
    1. Inadvertently including input prompt tokens in the output generation rate calculation.
    2. Averaging TPS across sessions without weighting by output length.
*   **How to collect it in Python**:
    ```python
    output_tokens = len(output_ids[0]) - len(input_ids[0])
    tps = output_tokens / (generation_time_ms / 1000)
    print(f"Throughput: {tps:.2f} tokens/sec")
    ```
*   **Expected ranges**: 
    *   *Single-User Local RTX 3090/4090 (8B Model)*: 45 to 85 tokens/sec.
    *   *Cloud serving system (Multi-user batched, H100)*: 1500+ aggregate tokens/sec.
*   **How it impacts user experience**: Directly correlates with text readability. Under 10 tokens/sec is hard to read in real time; above 30 tokens/sec is faster than human reading limits.
*   **How it impacts serving costs**: Directly proportional to revenue in commercial APIs. High TPS per GPU allows scaling to more users with fewer hardware nodes.
*   **How major inference systems optimize it**: Model quantization (AWQ, GPTQ), Flash Decoding, and specialized memory-bound kernels.

---

### 2. Prompt Throughput
*   **Definition**: The rate at which the system can process incoming prompt tokens per unit of time.
*   **Why it matters**: Key metric for search-based systems, long-document summarizers, and retrieval-augmented generation (RAG) where inputs are massive compared to outputs.
*   **How it is measured**: Divide total input prompt tokens processed by the total prefill time.
*   **Units**: Prompt tokens per second (tokens/sec).
*   **Common mistakes**: Combining prompt processing throughput with token generation throughput.
*   **How to collect it in Python**:
    ```python
    num_prompt_tokens = input_ids.shape[-1]
    prompt_throughput = num_prompt_tokens / (prefill_ms / 1000)
    print(f"Prompt Throughput: {prompt_throughput:.2f} tokens/sec")
    ```
*   **Expected ranges**: 10,000 to 150,000 tokens/sec on modern Enterprise GPUs (A100, H100).
*   **How it impacts user experience**: Impacts the speed at which long documents can be uploaded and initially queried.
*   **How it impacts serving costs**: Higher prompt throughput reduces the hardware footprint required to ingest large contextual data feeds.
*   **How major inference systems optimize it**: Chunked prefill, block-based prompt caching (e.g., RadixAttention in SGLang), and tensor parallelism.

---

### 3. Decode Throughput
*   **Definition**: The global rate of token generation across all active streams concurrently processed by the hardware.
*   **Why it matters**: Isolates the memory-bound workload efficiency of the system under multi-user concurrency.
*   **How it is measured**: Sum of all output tokens generated across all concurrent sessions divided by the duration of the monitoring window.
*   **Units**: Aggregate tokens per second (tokens/sec).
*   **Common mistakes**: Measuring decode throughput with a batch size of 1. It must be measured at saturating batch sizes.
*   **How to collect it in Python**:
    ```python
    # Simulated multi-stream tracker
    total_tokens_generated = sum(len(stream["outputs"]) for stream in active_streams)
    total_decode_time = monitoring_window_seconds
    aggregate_decode_tps = total_tokens_generated / total_decode_time
    ```
*   **Expected ranges**: 800 to 3,000 tokens/sec on enterprise server cards.
*   **How it impacts user experience**: Prevents degradation of individual streaming speed when other users submit requests simultaneously.
*   **How it impacts serving costs**: Represents the core efficiency ceiling. Maximizing decode throughput directly decreases cost-per-million-tokens.
*   **How major inference systems optimize it**: PagedAttention, Tensor Parallelism, and Tensor-Core-accelerated quantized GEMM kernels (e.g. FP8/INT4).

---

### 4. Requests Per Second (RPS)
*   **Definition**: The number of complete inference requests finished (or received) by the system per second.
*   **Why it matters**: The standard capacity planning metric for web services and server scaling.
*   **How it is measured**: Count total completed HTTP/gRPC requests over a fixed duration divided by the duration in seconds.
*   **Units**: Requests per second (Reqs/sec or RPS).
*   **Common mistakes**: Failing to report the average prompt and generation token lengths per request. RPS is meaningless without context on token counts.
*   **How to collect it in Python**:
    ```python
    import time
    
    start_time = time.time()
    # Run workload simulator...
    end_time = time.time()
    
    rps = total_completed_requests / (end_time - start_time)
    print(f"RPS: {rps:.2f} req/s")
    ```
*   **Expected ranges**: 0.5 to 50+ RPS per node depending on the model size and hardware.
*   **How it impacts user experience**: High RPS capacity ensures users do not hit "429 Too Many Requests" rate limits or queue timeouts during traffic spikes.
*   **How it impacts serving costs**: Higher RPS limits mean fewer servers are needed to handle the same user base.
*   **How major inference systems optimize it**: Dynamic load balancing, queuing, continuous batching, and aggressive model replication.

---

### 5. Concurrent Throughput
*   **Definition**: The total throughput achieved by the system when handling a fixed number of simultaneous active connections or clients.
*   **Why it matters**: Shows how throughput scales with load. Helps identify the concurrency level at which hardware performance saturates and latency starts to degrade exponentially.
*   **How it is measured**: Launch *N* client threads/processes sending continuous requests to the server, and calculate the global token generation rate.
*   **Units**: Aggregate tokens/sec at *N* concurrency.
*   **Common mistakes**: Measuring concurrency using artificial sleep loops inside client scripts instead of back-to-back request streams.
*   **How to collect it in Python**:
    Use asynchronous programming (`asyncio`) or thread pools to send concurrent requests:
    ```python
    import asyncio
    import aiohttp
    import time

    async def send_request(session, payload):
        start = time.perf_counter()
        async with session.post("http://localhost:8000/v1/completions", json=payload) as resp:
            data = await resp.json()
            latency = time.perf_counter() - start
            tokens = len(data["choices"][0]["text"].split()) # Approximate
            return tokens, latency

    async def main():
        payload = {"prompt": "Write a short poem", "max_tokens": 100}
        async with aiohttp.ClientSession() as session:
            # concurrency of 20
            tasks = [send_request(session, payload) for _ in range(20)]
            results = await asyncio.gather(*tasks)
            total_tokens = sum(r[0] for r in results)
            print(f"Aggregate tokens: {total_tokens}")
            
    # Run using asyncio.run(main())
    ```
*   **Expected ranges**: Under high concurrency (e.g., 64-128 streams), aggregate throughput on H100 can scale up to 3000 tokens/sec.
*   **How it impacts user experience**: High concurrent throughput keeps the queue short, ensuring prompt start times stay low during peak hours.
*   **How it impacts serving costs**: Determines hardware density metrics.
*   **How major inference systems optimize it**:
    *   **PagedAttention**: Prevents VRAM exhaustion when scaling concurrent users.
    *   **Continuous Batching**: Minimizes bubble size in pipeline execution.

---

## Part 3: Memory Metrics

Memory is the primary bottleneck in LLM inference. An LLM's weights, along with active user contexts, must fit within ultra-fast GPU memory (VRAM) to achieve acceptable speeds.

```
+-----------------------------------------------------------------------+
|                         VRAM Allocation Layout                        |
+-----------------------------------------------------------------------+
|                                                                       |
|  [Model Weights] (Static Size, e.g., ~16 GB for FP16 8B model)        |
|  * Persistent, loaded once                                            |
|                                                                       |
|  [KV Cache Allocation] (Dynamic, Managed by PagedAttention/Scheduler) |
|  * Stores Key/Value activations for all active tokens                 |
|                                                                       |
|  [Activation Memory] (Transient, Peak size during forward pass)       |
|  * Temporary tensors stored during backprop/forward steps             |
|                                                                       |
|  [Memory Fragmentation / Overhead] (Unusable VRAM gap)                |
|                                                                       |
+-----------------------------------------------------------------------+
```

### 1. RAM Usage
*   **Definition**: The quantity of standard system host memory consumed by the inference process.
*   **Why it matters**: High system RAM usage can lead to host paging/swapping to disk (which destroys performance) or out-of-memory (OOM) crashes on the CPU side.
*   **How it is measured**: Query the OS process information for Resident Set Size (RSS) memory.
*   **Units**: Gigabytes (GB).
*   **Common mistakes**: Monitoring Virtual Memory Size (VSZ) instead of Resident Set Size (RSS). VSZ includes mapped but unallocated space.
*   **How to collect it in Python**:
    ```python
    import psutil
    import os

    process = psutil.Process(os.getpid())
    ram_usage_gb = process.memory_info().rss / (1024 ** 3)
    print(f"Host RAM Usage: {ram_usage_gb:.2f} GB")
    ```
*   **Expected ranges**: 8 GB to 64 GB depending on the model weights stored in system RAM during CPU-offload or prior to GPU loading.
*   **How it impacts user experience**: Indirect. If host RAM is exhausted, the process crashes, interrupting all users.
*   **How it impacts serving costs**: CPU RAM is significantly cheaper than GPU VRAM. Optimizing RAM usage allows scaling host instances with cheaper configurations.
*   **How major inference systems optimize it**: Fast model loading via `mmap` (memory mapping) which loads weights dynamically as needed rather than reading everything into RAM at startup.

---

### 2. VRAM Usage
*   **Definition**: The amount of High Bandwidth Memory (HBM/VRAM) utilized on the GPU.
*   **Why it matters**: VRAM is the absolute physical limit of LLM serving. If a model and its KV cache exceed the available VRAM, the engine will crash with an out-of-memory (OOM) error or fall back to slow CPU execution.
*   **How it is measured**: Querying the CUDA runtime memory allocator or raw GPU driver statistics (via NVML).
*   **Units**: Gigabytes (GB).
*   **Common mistakes**: Querying `torch.cuda.memory_allocated()` and assuming it represents total GPU memory consumption. PyTorch maintains a private memory pool (caching allocator), so the physical memory allocated on the device is actually `torch.cuda.memory_reserved()`.
*   **How to collect it in Python**:
    ```python
    import torch

    # PyTorch-specific metrics (accurate for PyTorch operations)
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    ```
*   **Expected ranges**: 16 GB to 80+ GB per GPU.
*   **How it impacts user experience**: Indirect. Insufficient VRAM prevents using long context lengths or large batch sizes, degrading system responsiveness.
*   **How it impacts serving costs**: VRAM capacity dictates the maximum model size and maximum batch size that can be run on a single node. This is the single largest cost driver in LLM infrastructure.
*   **How major inference systems optimize it**: Quantization (reducing model footprint) and offloading KV caches to host RAM when active processing is idle.

---

### 3. Key-Value (KV) Cache Memory
*   **Definition**: Memory dedicated to saving key and value vectors of attention layers for historical tokens in active sequences, avoiding recalculation at each decode step.
*   **Why it matters**: The KV cache grows dynamically with context length and batch size. Its management determines the maximum batch size the server can handle.
*   **How it is measured**: Mathematically calculated or monitored via framework profiling.
    $$\text{Size (bytes)} = 2 \times (\text{layers}) \times (\text{attention heads}) \times (\text{head dimension}) \times (\text{precision bytes}) \times (\text{sequence length}) \times (\text{batch size})$$
*   **Units**: Gigabytes (GB).
*   **Common mistakes**: Underestimating KV cache size under long contexts. For a Llama-3-70B model with a batch size of 32 and 8k context length, the KV cache alone demands ~64 GB of VRAM.
*   **How to collect it in Python**:
    ```python
    # Theoretical KV Cache calculation for Llama-3-8B (FP16)
    layers = 32
    kv_heads = 8  # Grouped Query Attention
    head_dim = 128
    bytes_per_param = 2 # FP16
    context_len = 2048
    batch_size = 16

    kv_cache_bytes = 2 * layers * kv_heads * head_dim * bytes_per_param * context_len * batch_size
    kv_cache_gb = kv_cache_bytes / (1024 ** 3)
    print(f"Theoretical KV Cache Size: {kv_cache_gb:.2f} GB")
    ```
*   **Expected ranges**: 1 GB to 40+ GB.
*   **How it impacts user experience**: Ensures fast inter-token generation times by eliminating quadratic re-computation costs.
*   **How it impacts serving costs**: Directly controls the maximum batch size. Unoptimized KV caches lead to low batch sizes, increasing hardware cost per request.
*   **How major inference systems optimize it**:
    *   **Grouped-Query Attention (GQA)**: Sharing key/value projection heads across query heads to scale down KV cache size by 4x to 8x.
    *   **PagedAttention**: Organizing KV cache into non-contiguous blocks, eliminating memory waste from pre-allocating contiguous memory buffers.

---

### 4. Model Weight Memory
*   **Definition**: The static memory block allocated to hold the model parameter parameters (weights).
*   **Why it matters**: Determines the baseline hardware entry point. A model cannot run on a GPU unless its weights (or offloaded layers) fit into VRAM.
*   **How it is measured**:
    $$\text{Weight Memory} = \text{Parameter Count} \times \text{Bytes per Parameter}$$
*   **Units**: Gigabytes (GB).
*   **Common mistakes**: Forgetting to account for memory layout overheads or optimizer/runtime states if doing on-the-fly execution steps.
*   **How to collect it in Python**:
    ```python
    # Loop over parameters to sum up size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    weights_gb = (param_size + buffer_size) / (1024 ** 3)
    print(f"Model Weight memory: {weights_gb:.2f} GB")
    ```
*   **Expected ranges**: 
    *   *8B parameter model at FP16 (16-bit)*: ~16 GB.
    *   *8B parameter model at INT4 (4-bit)*: ~4.5 GB.
*   **How it impacts user experience**: Smaller weight sizes permit deploying highly complex models onto cheaper, local hardware, enabling fast execution speeds close to the user.
*   **How it impacts serving costs**: Directly defines the minimum GPU specification required.
*   **How major inference systems optimize it**: Quantization techniques such as AWQ, GPTQ, GGUF, and FP8 precision formats.

---

### 5. Activation Memory
*   **Definition**: Temporary memory consumed during the forward pass by intermediate tensor activations (outputs of layers, attention maps, dropout states).
*   **Why it matters**: Activation memory spikes during the forward pass and can trigger unexpected OOMs during prefill steps if not managed carefully.
*   **How it is measured**: Typically tracked using memory profilers (e.g., PyTorch Profiler or PyTorch memory snapshot).
*   **Units**: Megabytes (MB) or Gigabytes (GB).
*   **Common mistakes**: Assuming activation memory remains constant during the decode phase. It varies according to input context sequence length.
*   **How to collect it in Python**:
    ```python
    import torch

    torch.cuda.reset_peak_memory_stats()
    # Execute single forward pass
    with torch.no_grad():
        _ = model(input_ids)
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Peak VRAM during forward pass: {peak_vram:.2f} GB")
    ```
*   **Expected ranges**: 100 MB to 5 GB depending on model block architecture and attention type.
*   **How it impacts user experience**: Indirect.
*   **How it impacts serving costs**: High activation memory reduces the headroom available for the KV cache, shrinking the maximum batch size.
*   **How major inference systems optimize it**:
    *   **Activation Checkpointing (during training)**.
    *   **Kernel Fusion**: Fusing layers (e.g., RMSNorm + Linear) to process calculations within GPU SRAM registers without writing intermediate activations back to high-bandwidth VRAM.

---

### 6. Memory Fragmentation
*   **Definition**: The occurrence of unusable gaps in physical VRAM due to allocation and deallocation of non-contiguous, variable-sized chunks of memory.
*   **Why it matters**: A system can crash with an OOM even when total free memory is higher than the requested allocation if the free memory is split into tiny, non-contiguous blocks.
*   **How it is measured**: The difference between memory reserved by the allocator and active memory in use, divided by the reserved memory.
*   **Units**: Percentage (%).
*   **Common mistakes**: Ignoring memory fragmentation ratios when designing long-running production workloads.
*   **How to collect it in Python**:
    ```python
    import torch

    stats = torch.cuda.memory_stats()
    allocated = stats["allocated_bytes.all.current"]
    reserved = stats["reserved_bytes.all.current"]
    
    fragmentation_pct = (1.0 - (allocated / reserved)) * 100 if reserved > 0 else 0.0
    print(f"Memory Fragmentation: {fragmentation_pct:.2f}%")
    ```
*   **Expected ranges**: 5% to 40% (high fragmentation is standard under native PyTorch dynamic allocation).
*   **How it impacts user experience**: Can lead to unpredictable request failures mid-conversation.
*   **How it impacts serving costs**: Wastes up to a third of expensive GPU VRAM by rendering it unusable for KV caching.
*   **How major inference systems optimize it**:
    *   **Pre-allocation**: vLLM pre-allocates up to 90%+ of remaining free VRAM for the KV cache at startup, bypassing PyTorch's dynamic allocator and completely eliminating dynamic fragmentation.

---

## Part 4: GPU Metrics

GPU utilization profiling is key to confirming that the underlying parallel compute engine is being saturated efficiently.

### 1. GPU Utilization
*   **Definition**: The percentage of time over the past sample period during which one or more kernels were executing on the GPU.
*   **Why it matters**: A general indicator of whether the GPU is active. However, a high GPU utilization does not mean the hardware is running efficiently; it simply indicates that the GPU is not idle.
*   **How it is measured**: Sampled from OS-level driver queries via NVML.
*   **Units**: Percentage (%).
*   **Common mistakes**: Treating 100% GPU utilization as a sign of optimized code. A memory-bound operation waiting for memory transfers will report high GPU utilization despite achieving less than 5% of peak compute FLOPS.
*   **How to collect it in Python**:
    ```python
    from pynvml import *

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    util = nvmlDeviceGetUtilizationRates(handle)
    print(f"GPU Utilization: {util.gpu}%")
    nvmlShutdown()
    ```
*   **Expected ranges**: 0% (idle) to 100% (saturated).
*   **How it impacts user experience**: Higher efficiency translates to faster overall processing and lower latency per token under high loads.
*   **How it impacts serving costs**: Underutilization means wasted hardware expenditure.
*   **How major inference systems optimize it**: Batch size scaling, kernel compilation (PyTorch 2.0 compile), and continuous batching.

---

### 2. SM Utilization
*   **Definition**: The percentage of Streaming Multiprocessors (SMs) on the GPU that are actively executing instruction warps during a given sample window.
*   **Why it matters**: Provides a more granular look at compute saturation than generic GPU utilization. It shows if parallel blocks are actively executing code.
*   **How it is measured**: Collected via hardware performance monitors (using profiling APIs like NVIDIA Nsight or CUPTI).
*   **Units**: Percentage (%).
*   **Common mistakes**: Believing that high SM utilization correlates to high arithmetic efficiency.
*   **How to collect it in Python**: Requires NVLink/CUPTI profiler bindings or running external profilers like `ncu` (Nsight Compute).
*   **Expected ranges**: 30% (unbatched decode) to 98% (dense batched prefill).
*   **How it impacts user experience**: Higher SM utilization enables high-concurrency throughput scaling.
*   **How it impacts serving costs**: Maximizing SM usage optimizes throughput-per-dollar.
*   **How major inference systems optimize it**: Optimizing thread-block dimensions and maximizing warp occupancy.

---

### 3. Tensor Core Utilization
*   **Definition**: The percentage of active execution cycles in which the GPU's dedicated Tensor Cores (specialized matrix math units) are running matrix multiply-accumulate (MMA) instructions.
*   **Why it matters**: LLMs are built on matrix multiplications. Tensor Cores provide up to 16x higher throughput than standard CUDA FP32 units. If Tensor Core utilization is low, the model is not exploiting the hardware's primary compute engine.
*   **How it is measured**: Monitored via hardware performance counters (e.g., `tensor_precision_fu_utilization` via Nsight Compute).
*   **Units**: Percentage (%).
*   **Common mistakes**: Assuming Tensor Cores are active during standard FP32 operations. They require half-precision (FP16, BF16, FP8, or quantized formats) to activate.
*   **How to collect it in Python**:
    Must be profiled externally using NVIDIA Nsight Compute:
    ```bash
    ncu --metrics sm__pipe_tensor_op_hmma_cycle_active_real_power.avg.pct_of_peak_sustained_active <python_script.py>
    ```
*   **Expected ranges**: 5% (decode phase) to 60%+ (optimized prefill phase).
*   **How it impacts user experience**: High Tensor Core utilization reduces prefill times (TTFT) by multiple orders of magnitude.
*   **How it impacts serving costs**: Tensor Cores maximize compute efficiency, dramatically lowering electrical power and system footprint requirements.
*   **How major inference systems optimize it**: Ensuring matrix dimensions are multiples of 8 or 16 to meet Tensor Core alignment constraints, and using mixed-precision configurations.

---

### 4. Memory Bandwidth Utilization
*   **Definition**: The percentage of the GPU's maximum theoretical memory access bandwidth (speed at which data can be read from or written to HBM/VRAM) currently being utilized.
*   **Why it matters**: The decode phase of LLMs is entirely memory-bandwidth bound. Optimizing memory bandwidth utilization is key to accelerating individual token generation.
*   **How it is measured**: Calculated from memory read/write counters or queried via NVML.
    $$\text{Bandwidth Utilized} = \frac{\text{Bytes Read} + \text{Bytes Written}}{\text{Time} \times \text{Theoretical Peak Bandwidth}}$$
*   **Units**: Percentage (%) or Gigabytes per Second (GB/s).
*   **Common mistakes**: Ignoring memory throughput limits when designing model architectures (e.g., selecting architectures with excessive parameters relative to attention heads).
*   **How to collect it in Python**:
    ```python
    from pynvml import *
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    util = nvmlDeviceGetUtilizationRates(handle)
    print(f"Memory Controller Utilization: {util.memory}%")
    nvmlShutdown()
    ```
*   **Expected ranges**: 60% to 95% in optimized decode pipelines.
*   **How it impacts user experience**: Directly determines the streaming tokens per second for a single user.
*   **How it impacts serving costs**: Higher bandwidth efficiency allows more concurrent requests to be handled before latency degrades.
*   **How major inference systems optimize it**: Weight quantization (reduces weight byte transfers) and FlashAttention (minimizes writebacks of intermediate attention matrices to VRAM).

---

### 5. PCIe Transfer Overhead
*   **Definition**: The time spent transferring model weights, input tokens, or output logits across the PCIe bus between the host CPU RAM and the GPU VRAM.
*   **Why it matters**: The PCIe bus is several orders of magnitude slower than the GPU's internal HBM memory bus. Frequent transfers between host and device create severe execution bottlenecks.
*   **How it is measured**: Profiling Host-to-Device (H2D) and Device-to-Host (D2H) copy calls.
*   **Units**: Milliseconds (ms) or Megabytes per Second (MB/s).
*   **Common mistakes**: Initiating tokenization on CPU and transferring intermediate tensor structures inside the generation loop instead of caching state on-device.
*   **How to collect it in Python**:
    ```python
    import time
    import torch

    tensor_cpu = torch.randn(10000, 10000)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    # Measure transfer speed
    tensor_gpu = tensor_cpu.cuda()
    
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    mb_transferred = (tensor_cpu.nelement() * tensor_cpu.element_size()) / (1024 ** 2)
    print(f"PCIe H2D Transfer Speed: {mb_transferred / duration:.2f} MB/s")
    ```
*   **Expected ranges**: 6 GB/s to 12 GB/s on PCIe Gen3; 12 GB/s to 26 GB/s on PCIe Gen4; up to 50+ GB/s on PCIe Gen5.
*   **How it impacts user experience**: High PCIe transfers cause delays in TTFT and introduce stuttering during decoding.
*   **How it impacts serving costs**: Limits the efficiency of multi-GPU systems if inter-GPU transfer relies on PCIe instead of NVLink.
*   **How major inference systems optimize it**: Pinning host memory (`pin_memory()`) to enable asynchronous, non-blocking transfers, and processing tokenization and sampling entirely on the GPU.

---

## Part 5: CPU Metrics

Even when using GPU acceleration, the CPU controls host scheduling, model orchestration, dynamic input queuing, and networking.

### 1. CPU Utilization
*   **Definition**: The percentage of CPU cores actively executing system processes.
*   **Why it matters**: In hybrid execution environments (e.g., GGUF split offloading), the CPU handles a portion of the model layers. If the CPU is saturated, the entire inference pipeline stalls.
*   **How it is measured**: Monitored via OS schedulers.
*   **Units**: Percentage (%).
*   **Common mistakes**: Measuring global CPU utilization instead of tracking usage per logical thread, which can obscure single-thread bottleneck issues.
*   **How to collect it in Python**:
    ```python
    import psutil
    print(f"System CPU Utilization: {psutil.cpu_percent()}%")
    print(f"Per-core CPU Utilization: {psutil.cpu_percent(percpu=True)}")
    ```
*   **Expected ranges**: 5% to 20% during GPU-only inference; up to 100% during CPU offload.
*   **How it impacts user experience**: High host CPU utilization causes latency spikes by delaying GPU instruction dispatching.
*   **How it impacts serving costs**: Underutilized CPUs represent wasted system budget; over-specifying host CPUs adds unnecessary cost.
*   **How major inference systems optimize it**: Offloading tokenization, input validation, and sampling logic to high-performance C++ runtimes rather than Python.

---

### 2. Context Switching
*   **Definition**: The kernel-level procedure of saving and restoring the execution state of CPU threads to share physical execution cores.
*   **Why it matters**: Excessive context switching indicates thread contention, which wastes CPU cache lines and introduces scheduling latency.
*   **How it is measured**: Sampled from OS process status registers.
*   **Units**: Absolute number of context switches per second.
*   **Common mistakes**: Setting thread pool sizes (e.g., in OpenMP or MKL) higher than the physical core count, triggering thread thrashing.
*   **How to collect it in Python**:
    ```python
    import psutil
    import os

    proc = psutil.Process(os.getpid())
    switches = proc.num_ctx_switches()
    print(f"Voluntary switches: {switches.voluntary}, Involuntary: {switches.involuntary}")
    ```
*   **Expected ranges**: Under 10,000/sec on a well-tuned system.
*   **How it impacts user experience**: Causes latency spikes and increases variance (jitter) in TTFT.
*   **How it impacts serving costs**: Wastes CPU cycles on scheduling overhead rather than processing payload traffic.
*   **How major inference systems optimize it**: Setting thread affinity, using lock-free message queues, and tuning the thread pool to match physical cores.

---

### 3. Thread Efficiency
*   **Definition**: The ratio of active execution time to total lifetime for engine threads.
*   **Why it matters**: Low thread efficiency indicates that threads are frequently blocked waiting for locks, I/O, or GPU synchronization.
*   **How it is measured**: Profiling thread execution states (running, sleeping, blocked).
*   **Units**: Percentage (%).
*   **Common mistakes**: Using a single thread pool to handle both network I/O and GPU model execution.
*   **How to collect it in Python**:
    Must be profiled using tools like PySpy, Intel VTune, or native profiling frameworks.
*   **Expected ranges**: 85% to 98% in highly optimized engines.
*   **How it impacts user experience**: Improves the throughput stability and responsiveness of the application.
*   **How it impacts serving costs**: Wasted thread efficiency requires scaling up instance sizes prematurely.
*   **How major inference systems optimize it**: Decoupling the network input server (e.g., FastAPI/uvicorn loop) from the execution engine worker threads using async queues.

---

### 4. NUMA Considerations
*   **Definition**: Non-Uniform Memory Access (NUMA) is a computer memory design used in multiprocessor systems where memory access time depends on the memory location relative to the processor.
*   **Why it matters**: If a thread running on CPU Socket 0 accesses memory allocated in Socket 1's memory domain, the transfer must cross the inter-socket bus, which significantly increases latency.
*   **How it is measured**: Queried using NUMA configuration tools (`numactl` on Linux).
*   **Units**: Access latency penalties (nanoseconds).
*   **Common mistakes**: Running multi-socket server nodes without pinning threads and memory to specific sockets, resulting in variable benchmarking numbers.
*   **How to collect it in Python**:
    Python standard library does not handle NUMA pinning directly. Pinning is typically configured at the OS level:
    ```bash
    # Pin inference worker to NUMA node 0 and its local memory
    numactl --cpunodebind=0 --membind=0 python inference_server.py
    ```
*   **Expected ranges**: Cross-socket memory transfers can introduce a 1.5x to 2.5x latency penalty.
*   **How it impacts user experience**: Under high loads, NUMA imbalances cause erratic spikes in generation speed.
*   **How it impacts serving costs**: Reduces the maximum throughput capacity of large multi-socket server configurations.
*   **How major inference systems optimize it**: Configuring single-NUMA affinity groups per GPU instance, pinning CPU cores, and splitting weights across separate processes mapped to local memory channels.

---

## Part 6: Model Metrics

Model metrics define the static architecture and runtime configurations of the neural network itself.

### 1. Parameter Count
*   **Definition**: The total number of weights (parameters) configured in the neural network layers.
*   **Why it matters**: The fundamental sizing metric of a model. It determines the base compute footprint (FLOPs per token) and memory storage footprint.
*   **How it is measured**: Summing the size of all parameter tensors in the model.
*   **Units**: Billions of parameters (B).
*   **Common mistakes**: Stating parameter count without clarifying if it refers to total parameters or active parameters (in Mixture of Experts models like Mixtral, active parameters per token are a fraction of the total parameters).
*   **How to collect it in Python**:
    ```python
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e9:.2f} Billion")
    ```
*   **Expected ranges**: 1B to 400B+.
*   **How it impacts user experience**: Larger models generally exhibit higher reasoning quality, but run slower and require more powerful hardware.
*   **How it impacts serving costs**: Scales memory and compute requirements linearly or quadratically.
*   **How major inference systems optimize it**: Mixture of Experts (MoE) architectures, which activate only a subset of parameters per token to reduce execution compute requirements.

---

### 2. Quantization Type
*   **Definition**: The precision format used to store model weights and activations (e.g., FP32, FP16, BF16, FP8, INT8, INT4).
*   **Why it matters**: Quantization reduces weight storage requirements and speeds up memory transfer times, enabling larger models to run on cheaper hardware.
*   **How it is measured**: Inspected from tensor metadata definitions.
*   **Units**: Precision bits per weight (e.g., 4-bit, 8-bit, 16-bit).
*   **Common mistakes**: Assuming all quantization algorithms perform identically. GGUF, AWQ, GPTQ, and EXL2 use different formats that trade off execution speed and perplexity.
*   **How to collect it in Python**:
    ```python
    # Check data type of first linear layer parameter
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"Precision type: {param.dtype}")
            break
    ```
*   **Expected ranges**: FP16/BF16 down to INT4/FP8.
*   **How it impacts user experience**: Lower bit-widths increase generation speed, but extreme quantization (e.g., 2-bit) causes output degradation or repetitive loops.
*   **How it impacts serving costs**: Using INT4 instead of FP16 reduces the GPU count required to serve a model by 4x.
*   **How major inference systems optimize it**: Implementing custom CUDA kernels (e.g., Marlin, AWQ kernels) that unpack quantized weights into SRAM registers on-the-fly to execute high-speed GEMM math.

---

### 3. Context Length
*   **Definition**: The maximum number of combined input and output tokens a model can process in a single inference session.
*   **Why it matters**: Determines the maximum sequence length the model can process, but also defines the peak size of the KV cache.
*   **How it is measured**: Configured in the model's hyperparameter registry (e.g., `max_position_embeddings`).
*   **Units**: Tokens.
*   **Common mistakes**: Attempting to run context lengths beyond the model's calibrated limits without modifying RoPE positional scaling configs, resulting in gibberish output.
*   **How to collect it in Python**:
    ```python
    context_length = model.config.max_position_embeddings
    print(f"Model Max Context Length: {context_length} tokens")
    ```
*   **Expected ranges**: 2,048 (older models) to 1,040,000+ (e.g., Gemini).
*   **How it impacts user experience**: Enables deep analysis of long documents, entire codebases, or extended chat histories.
*   **How it impacts serving costs**: Scales KV cache memory footprint linearly or quadratically, which can quickly exhaust VRAM.
*   **How major inference systems optimize it**: FlashAttention-2, Flash Decoding, RoPE scaling adjustments, and sparse attention mechanisms.

---

### 4. Batch Size
*   **Definition**: The number of independent requests processed simultaneously in a single forward pass of the model.
*   **Why it matters**: The primary tool for maximizing throughput. Higher batch sizes allow memory-bound decode operations to run closer to compute-bound efficiency.
*   **How it is measured**: Checked at the engine scheduler level.
*   **Units**: Integer (batch count).
*   **Common mistakes**: Setting the batch size too high, causing the system to run out of memory (OOM) during prompt spikes.
*   **How to collect it in Python**:
    ```python
    # Input tensor batch dimension
    batch_size = input_ids.shape[0]
    print(f"Active Batch Size: {batch_size}")
    ```
*   **Expected ranges**: 1 (local/testing) to 256+ (large-scale serving).
*   **How it impacts user experience**: Increasing batch size can slightly increase individual inter-token latency, but increases system capacity so that new requests start processing faster.
*   **How it impacts serving costs**: Dramatically lowers serving cost per user by maximizing GPU core utilization.
*   **How major inference systems optimize it**:
    *   **Continuous/Dynamic Batching**: Merging incoming requests into active execution batches on the fly.

---

### 5. Number of GPU Layers (Offloading Count)
*   **Definition**: The number of transformer blocks (layers) pinned and executed on the GPU, while the remaining layers are processed by system memory and the CPU.
*   **Why it matters**: In hybrid execution architectures (e.g., llama.cpp running GGUF models), splitting layers allows running models that are larger than the available VRAM.
*   **How it is measured**: Inspected from model loading configurations.
*   **Units**: Integer count of layers.
*   **Common mistakes**: Setting this number without considering the KV cache size, which can result in OOM errors as context grows.
*   **How to collect it in Python**:
    In llama-cpp-python:
    ```python
    # Pass 'n_gpu_layers' to Llama constructor
    from llama_cpp import Llama
    llm = Llama(model_path="model.gguf", n_gpu_layers=24) # 24 layers offloaded to GPU
    ```
*   **Expected ranges**: 0 (all CPU) to full model layers (e.g., 32 layers for Llama-3-8B).
*   **How it impacts user experience**: Every layer running on the CPU significantly reduces generation speed.
*   **How it impacts serving costs**: Enables utilizing cheaper GPU configurations by leveraging system RAM.
*   **How major inference systems optimize it**: Optimizing CPU-GPU transfer channels, using SIMD extensions (AVX-512, AMX), and pinning memory blocks to reduce transfer overhead.

---

## Part 7: Quality vs. Performance Metrics

Optimizing inference often involves balancing hardware performance against model response quality.

```
+-----------------------------------------------------------------------+
|                       Performance-Quality Tradeoff                    |
+-----------------------------------------------------------------------+
|                                                                       |
|  [FP16 / Native Precision]                                            |
|  * High Perplexity (Best Quality), Large VRAM Footprint, Low TPS      |
|                                                                       |
|  [Quantized (e.g., INT4 / FP8)]                                       |
|  * Marginal Perplexity loss, Small VRAM Footprint, High TPS           |
|                                                                       |
+-----------------------------------------------------------------------+
```

### 1. Accuracy vs. Quantization (Perplexity)
*   **Definition**: The mathematical relationship between precision reduction (quantization) and model reasoning accuracy (typically evaluated using WikiText perplexity).
*   **Why it matters**: Quantizing a model down to 4-bit or 2-bit saves memory and accelerates generation, but degrades the model's reasoning capabilities and language coherence.
*   **How it is measured**: Evaluate the quantized model on standard benchmarks (MMLU, GSM8K, WikiText-2 perplexity) and compare scores against the FP16 base model.
*   **Units**: Perplexity (lower is better) or accuracy percentages.
*   **Common mistakes**: Assuming a quantization method behaves uniformly across different model architectures. Models under 8B parameters degrade much faster under quantization than 70B+ models.
*   **How to collect it in Python**:
    Use libraries like Hugging Face `evaluate` or evaluate on raw datasets like WikiText:
    ```python
    # Pseudocode for perplexity calculation
    import torch
    
    def calculate_perplexity(model, encodings):
        max_length = model.config.max_position_embeddings
        stride = 512
        seq_len = encodings.input_ids.size(1)
        
        nlls = []
        for i in range(0, seq_len, stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, seq_len)
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
                
            nlls.append(neg_log_likelihood)
            
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        return ppl.item()
    ```
*   **Expected ranges**: Perplexity delta under 0.1 is considered optimal for quantized models.
*   **How it impacts user experience**: Negligible for 4-bit/8-bit models; 2-bit models will hallucinate frequently or output repetitive loops.
*   **How it impacts serving costs**: INT4/FP8 quantization reduces VRAM requirements by 2-4x, allowing models to be served on significantly cheaper hardware configurations.
*   **How major inference systems optimize it**:
    *   **AWQ (Activation-aware Weight Quantization)**: Protecting the 1% most important weights during quantization to preserve accuracy.
    *   **SmoothQuant**: Smoothing activation outliers to enable robust INT8 quantization.

---

### 2. Latency vs. Throughput (The Server Frontier)
*   **Definition**: The tradeoff between single-user response speed (latency) and global request capacity (throughput).
*   **Why it matters**: As batch size increases, global system throughput improves, but individual user latency increases due to queuing and step overhead.
*   **How it is measured**: Measure the system at different batch sizes, plotting latency on the Y-axis and throughput on the X-axis to find the optimal operating point.
*   **Units**: Latency (ms) vs. Throughput (tokens/sec).
*   **Common mistakes**: Running production systems at maximum throughput limits, which can cause user request latencies to spike beyond acceptable thresholds.
*   **How to collect it in Python**:
    Benchmark the engine under varying batch sizes ($1, 2, 4, 8, 16, 32, 64$) and record both TTFT/ITL and aggregate TPS.
*   **Expected ranges**: 
    ```
    Batch=1  -> Latency: 15ms/token, Throughput: 65 tokens/sec
    Batch=32 -> Latency: 32ms/token, Throughput: 1000 tokens/sec
    ```
*   **How it impacts user experience**: Directly determines the response delay users experience during peak traffic periods.
*   **How it impacts serving costs**: Finding the optimal tradeoff point is key to maximizing hardware utilization while meeting service level agreements (SLAs).
*   **How major inference systems optimize it**: Dynamic scheduling algorithms that adjust batch sizes on-the-fly based on current queue lengths and target latency limits.

---

### 3. Memory vs. Speed
*   **Definition**: The performance tradeoffs involved in storing data in memory vs. recomputing it.
*   **Why it matters**: Caching states (like the KV cache) speeds up inference but consumes large amounts of VRAM, which limits the active batch size.
*   **How it is measured**: Comparing performance under different caching configurations (e.g., streaming LLM execution with no cache, partial cache, or full cache).
*   **Units**: VRAM (GB) vs. Tokens per Second (TPS).
*   **Common mistakes**: Believing that enabling all caching options always improves performance. If cache size forces the engine to swap data to host memory, performance will collapse.
*   **How to collect it in Python**:
    Measure total generation time with `use_cache=True` vs. `use_cache=False`.
*   **Expected ranges**: Disabling the KV cache can drop generation throughput by 5-10x for long sequences.
*   **How it impacts user experience**: Enabling caching is critical for maintaining fast generation speeds in conversational applications.
*   **How it impacts serving costs**: Caching requires careful VRAM management to avoid scaling up hardware costs unnecessarily.
*   **How major inference systems optimize it**:
    *   **PagedAttention**: Organizes cache allocations into small blocks to eliminate memory fragmentation.
    *   **GQA (Grouped Query Attention)**: Reduces the size of KV cache entries by sharing key/value states across query heads.

---

### 4. Cost vs. Quality
*   **Definition**: The trade-offs involved in selecting a model size and hosting infrastructure relative to the quality of the generated outputs.
*   **Why it matters**: Large models (e.g., 70B+) provide higher reasoning quality but are expensive to serve, whereas smaller models (e.g., 8B) are fast and cheap but may fail complex reasoning tasks.
*   **How it is measured**:
    $$\text{Cost-Efficiency Metric} = \frac{\text{Task Evaluation Accuracy}}{\text{Serving Cost per 1M Tokens}}$$
*   **Units**: Accuracy % per Dollar.
*   **Common mistakes**: Defaulting to the largest available model for simple text processing tasks, resulting in unnecessary serving costs.
*   **How to collect it in Python**:
    Compute API usage billing logs against benchmark evaluation accuracy runs.
*   **Expected ranges**: Deploying a quantized 8B model can be 10-50x cheaper than serving a full FP16 70B model.
*   **How it impacts user experience**: Balances response speed and quality; using a model that is too small can lead to low-quality outputs, while a model that is too large can result in slow response times.
*   **How it impacts serving costs**: This is the primary business metric for commercial LLM deployments.
*   **How major inference systems optimize it**: Using routing layers (router models) that analyze incoming queries and dispatch them to the smallest/cheapest model capable of handling the request.

---

## Part 8: Benchmark Design Methodology

Designing benchmarks that yield consistent, reproducible, and unbiased results requires controlling for several hardware and software variables.

### How to Design Fair Benchmarks
1.  **Define Realistic Workloads**: Use prompts and output lengths that mimic actual user queries rather than relying on uniform, artificial sizes (e.g., 128 input tokens, 128 output tokens).
2.  **Isolate the System Under Test**: Terminate all non-essential host background processes, disconnect network links if testing locally, and lock GPU clock speeds to eliminate thermal throttling variables.
3.  **Perform Warm-up Runs**: Modern runtimes compile code and cache memory blocks during the first few requests. Always discard the first 3-5 iterations to ensure metrics reflect steady-state performance.
4.  **Control Cache Effects**: Clear system memory caches and reset GPU allocators between tests to prevent prompt caching from artificially inflating results.
5.  **Ensure Statistical Significance**: Run each test case across multiple iterations ($N \ge 30$) and report statistical distributions (mean, median, p95, p99) rather than relying on single-run data.

```
+-----------------------------------------------------------------------+
|                    Benchmarking Latency Distribution                  |
+-----------------------------------------------------------------------+
|                                                                       |
|  [p50 / Median Latency]                                               |
|  * Represents typical user experience                                 |
|                                                                       |
|  [p95 / p99 Latency]                                                  |
|  * Captures worst-case spikes (e.g., queue stalls, garbage collection)|
|                                                                       |
+-----------------------------------------------------------------------+
```

### Statistical Significance and Confidence Intervals
When comparing two systems, calculate the Confidence Interval (CI) to determine if performance differences are statistically significant or just noise.
*   **Formula for the 95% Confidence Interval**:
    $$\text{CI} = \bar{x} \pm t^* \left( \frac{s}{\sqrt{n}} \right)$$
    Where:
    *   $\bar{x}$ = Sample mean.
    *   $t^*$ = Critical value from the $t$-distribution (approx. 1.96 for large sample sizes).
    *   $s$ = Sample standard deviation.
    *   $n$ = Number of runs.

```python
import numpy as np
import scipy.stats as stats

def calculate_95_ci(data):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data) # Standard error of the mean
    margin = sem * stats.t.ppf((1 + 0.95) / 2.0, n - 1)
    return mean - margin, mean + margin
```

---

## Part 9: Optimization & Experiment Roadmap (Project-First Curriculum)

This roadmap outlines a hands-on, systems-first engineering curriculum structured over 6 months. It focuses on building measurement tools, hitting real resource bottlenecks on target consumer hardware (specifically an NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM, 16 GB DDR5 RAM, and Intel Core i5 CPU), implementing caching algorithms, reading foundational optimization papers, and prototyping novel optimizations.

```
+-------------------------------------------------------------------------------------------------+
|                                    Inference Optimization Roadmap                               |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
| [Month 1: Build the Lab]     -> [Month 2: Transformer Internals] -> [Month 3: Paper & SnapKV]   |
| Weeks 1-2: llama.cpp setup      Weeks 5-6: Tiny GPT from scratch    Week 9: FlashAttention read |
| Weeks 3-4: Benchmark script     Weeks 7-8: KV Cache memory profiler Weeks 10-12: SnapKV impl    |
|                                                                                                 |
| [Month 4: Deepen the Lab]    -> [Month 5: Prototype Novel Idea]  -> [Month 6: Implement & Eval] |
| Weeks 13-14: PagedAttention/SD  Weeks 17-18: Research question      Weeks 21-23: Evaluation     |
| Weeks 15-16: llama.cpp tracing  Weeks 19-20: Rapid prototype        Week 24: Technical report   |
|                                                                                                 |
+-------------------------------------------------------------------------------------------------+
```

All generated deliverables are strictly routed to their designated directories:
*   **Theory Reference Files**: Saved in `docs/` using only the 14 predefined knowledge-base filenames (e.g., `docs/transformer-basics.md`, `docs/kv-cache.md`).
*   **Benchmark Reports**: Saved in `benchmarks/` following the `DD-MM-YYYY-experiment-name.md` format.
*   **Learning Reports**: Saved in `learnings/` using the `learning-*.md` format.
*   **Workload Databases**: Saved in `results/` (as `benchmark_history.csv` or JSON runs).
*   **Code Modules**: Root directory or component subfolders (e.g., `tiny_gpt/`, `snapkv/`).

---

### Month 1 — Build the Benchmark Lab

#### Weeks 1–2: Set Up llama.cpp & Run Your First Model
*   **Goal**: Install `llama.cpp` from source with CUDA support and establish a baseline execution environment using the Qwen2.5-7B-Instruct model.
*   **Target hardware decisions**: Calibrate the number of GPU offloaded layers (`-ngl` / `--n-gpu-layers`). Observe the throughput differences as layers are offloaded to VRAM vs. system RAM.
*   **Reference Materials**: 
    *   [llama.cpp GitHub Repository](https://github.com/ggerganov/llama.cpp)
    *   [Hugging Face GGUF Model Repository](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)
    *   [Ollama homepage](https://ollama.com)
*   **Walls Hit & Learnings**:
    *   *Wall: "what does -ngl actually do and why does changing it affect speed?"* - Learned about memory transfer bottlenecks (PCIe) and VRAM vs RAM bandwidth. Read [Tim Dettmers' GPU blog post](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/).
    *   *Wall: "why is my 6GB GPU not faster than CPU for some operations?"* - Read [JAX Scaling Book roofline chapter](https://jax-ml.github.io/scaling-book/roofline/) to distinguish memory-bound vs compute-bound workloads.
*   **Deliverables**:
    *   **Theory Reference**: Initialize [docs/benchmarking-methodology.md](benchmarking-methodology.md) (controls, thermal locks).
    *   **Benchmark Report**: [benchmarks/12-06-2026-baseline-benchmark.md](../benchmarks/12-06-2026-baseline-benchmark.md) (documenting partial vs full GPU offload baseline timings).
    *   **Learning Report**: [learnings/learning-hardware-setup.md](../learnings/learning-hardware-setup.md) detailing compilation issues, `-ngl` findings, and memory-bandwidth observations.

#### Weeks 3–4: Build the Benchmark Script
*   **Goal**: Create a reusable Python benchmark suite (`benchmark.py`) to systematically record key latency and resource metrics across different quantization levels (Q4_K_M, Q5_K_M, Q8_0, and F16).
*   **Metrics Recorded**: Tokens/sec, Time to First Token (TTFT), inter-token latency, CPU RAM, and GPU VRAM usage.
*   **Reference Materials**:
    *   [llama-cpp-python Repository](https://github.com/abetlen/llama-cpp-python)
    *   `psutil` and `GPUtil` documentation.
*   **Walls Hit & Learnings**:
    *   *Wall: "why does Q4 vs Q8 affect speed, what is actually different?"* - Read GGUF specifications and [Tim Dettmers' 8-bit quantization post](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) to understand GGUF block-wise quantization mechanisms.
    *   *Wall: "what is tokens/sec actually measuring and why does prompt length affect it?"* - Watched [3Blue1Brown's Linear Algebra series](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) to map matrix multiplication cost scaling.
*   **Deliverables**:
    *   **Code**: [benchmark.py](../benchmark.py), [utils/system_monitor.py](../utils/system_monitor.py), [utils/metrics.py](../utils/metrics.py), and [utils/logging.py](../utils/logging.py).
    *   **Workload Database**: [results/benchmark_history.csv](../results/benchmark_history.csv) (persisted history) and [results/json/](../results/json) (raw run details).
    *   **Theory Reference**: Initialize [docs/quantization.md](quantization.md) (precision formats theory).
    *   **Benchmark Report**: [benchmarks/15-06-2026-quantization-benchmark.md](../benchmarks/15-06-2026-quantization-benchmark.md) comparing speed and resource usage across quants.
    *   **Learning Report**: [learnings/learning-quantization-differences.md](../learnings/learning-quantization-differences.md) explaining why Q4_K_M is the default for a 6GB VRAM target.

---

### Month 2 — Transformer Internals & KV Cache Profiling

#### Weeks 5–6: Build a Tiny GPT
*   **Goal**: Write a simple autoregressive GPT from scratch in PyTorch to gain a hands-on understanding of self-attention mechanics, weight matrices, and tensor transitions.
*   **Reference Materials**:
    *   [Andrej Karpathy's "micrograd" tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0)
    *   [Andrej Karpathy's "GPT from scratch" tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY)
    *   [Jay Alammar's "Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/)
*   **Walls Hit & Learnings**:
    *   *Wall: "what is the attention matrix actually computing and why Q × Kᵀ?"* - Read [Lilian Weng's "Attention? Attention!" post](https://lilianweng.github.io/posts/2018-06-24-attention/) to understand key-value similarity measurements.
    *   *Wall: "why does attention scale as O(n²) and why does that matter for inference speed?"* - Tested prompt scaling limits in the benchmark lab and plotted latency to visualize the quadratic curve.
*   **Deliverables**:
    *   **Code**: Create `tiny_gpt/model.py`, `tiny_gpt/train.py`, and `tiny_gpt/cache_comparison.py`.
    *   **Theory Reference**: Initialize [docs/transformer-basics.md](transformer-basics.md) detailing multi-head attention and architecture shapes.
    *   **Learning Report**: [learnings/learning-transformer-internals.md](../learnings/learning-transformer-internals.md) mapping transformer weights, dynamic caching overhead, and context limit mathematics.

#### Weeks 7–8: Build a KV Cache Memory Profiler
*   **Goal**: Measure and model the memory growth of the KV cache across varying context lengths, determining at what point the system runs out of VRAM headroom.
*   **Reference Materials**:
    *   [Lilian Weng's "LLM Inference Optimization" post](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) (specifically the KV cache calculations).
*   **Walls Hit & Learnings**:
    *   *Wall: "what exactly is stored in the KV cache and how do I calculate its size?"* - Modeled the allocation formula: $2 \times \text{layers} \times \text{heads} \times \text{head\_dim} \times \text{context\_len} \times \text{bytes\_per\_element}$ and validated it against measured GPU allocations.
*   **Deliverables**:
    *   **Code**: Create `kv_cache_profiler.py`.
    *   **Workload Database**: Save metrics to [results/kv_cache_growth.csv](../results/kv_cache_growth.csv).
    *   **Theory Reference**: Initialize [docs/kv-cache.md](kv-cache.md) detailing sizing mathematics and allocation logic.
    *   **Benchmark Report**: [benchmarks/25-06-2026-kv-cache-profiler.md](../benchmarks/25-06-2026-kv-cache-profiler.md) plotting the memory consumption profile of baseline Qwen models.
    *   **Learning Report**: [learnings/learning-kv-cache-math.md](../learnings/learning-kv-cache-math.md) detailing the VRAM boundaries, allocations, and how bandwidth constraints emerge.

---

### Month 3 — Paper Reading & SnapKV Implementation

#### Week 9: Learn How to Read a Paper & Study FlashAttention
*   **Goal**: Master academic paper analysis techniques and read the original FlashAttention paper to understand SRAM tiling and IO memory-bound bottlenecks.
*   **Reference Materials**:
    *   [S. Keshav's "How to Read a Paper" guidelines](http://ccr.sigcomm.org/online/files/p83-keshavA.pdf)
    *   [Aleksa Gordić's "ELI5 FlashAttention" breakdown](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
    *   [FlashAttention Paper (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
*   **Deliverables**:
    *   **Theory Reference**: Initialize [docs/flash-attention.md](flash-attention.md) outlining fused kernels and SRAM tiles.
    *   **Learning Report**: [learnings/learning-flashattention-memory-io.md](../learnings/learning-flashattention-memory-io.md) detailing hardware-level analyses of HBM transfers vs kernel processing.

#### Weeks 10–12: Implement SnapKV and Benchmark It
*   **Goal**: Build a custom PyTorch/Transformers hook that compresses the KV cache dynamically using SnapKV (clustering key-value pairs by attention scores) and evaluate its speed vs. perplexity trade-offs.
*   **Reference Materials**:
    *   [SnapKV Paper (Li et al., 2024)](https://arxiv.org/abs/2404.14469)
    *   [SnapKV GitHub Repository](https://github.com/FasterDecoding/SnapKV)
*   **Walls Hit & Learnings**:
    *   *Wall: "how do I hook into the attention layer to intercept the KV cache?"* - Explored Hugging Face Transformers' internal state management and intercepted the `past_key_values` object.
    *   *Wall: "my perplexity is much worse than the paper reports — why?"* - Ran a comprehensive parameter sweep on $K$ (cache size limit) from 16 to 256 to isolate the optimal threshold.
*   **Deliverables**:
    *   **Code**: Create `snapkv/hook.py` and `snapkv/eval.py`.
    *   **Workload Database**: Save sweeps to `results/snapkv_benchmark.csv`.
    *   **Theory Reference**: Update [docs/kv-cache.md](kv-cache.md) adding attention pooling and compression algorithms.
    *   **Benchmark Report**: [benchmarks/15-07-2026-snapkv-compression.md](../benchmarks/15-07-2026-snapkv-compression.md) showing VRAM savings and quality (perplexity) parameters over sweeps.
    *   **Learning Report**: [learnings/learning-snapkv-mechanics.md](../learnings/learning-snapkv-mechanics.md) covering hook insertions, key retention metrics, and consumer GPU execution efficiency.

---

### Month 4 — PagedAttention, Speculative Decoding & llama.cpp Tracing

#### Weeks 13–14: Read PagedAttention & Speculative Decoding
*   **Goal**: Study virtual memory page allocations for KV cache blocks (PagedAttention) and speculative generation architectures.
*   **Reference Materials**:
    *   [vLLM Blog Post on PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
    *   [Lilian Weng's Speculative Decoding explanation](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
    *   [Speculative Decoding Paper (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192)
*   **Deliverables**:
    *   **Theory Reference**: Initialize [docs/paged-attention.md](paged-attention.md) (block mapping layouts) and [docs/speculative-decoding.md](speculative-decoding.md) (acceptance criteria/draft networks).
    *   **Learning Report**: [learnings/learning-speculative-decoding-constraints.md](../learnings/learning-speculative-decoding-constraints.md) highlighting VRAM capacity limits and latency tradeoffs during dual-model serving on consumer chips.

#### Weeks 15–16: Read llama.cpp Source & Measure Speculative Decoding
*   **Goal**: Trace the inner inference loop and KV cache management in the `llama.cpp` codebase, and benchmark speculative decoding natively using a tiny draft model (e.g., Qwen2.5-1B) paired with the baseline model.
*   **Reference Materials**:
    *   [llama.cpp sampling and execution loops](https://github.com/ggerganov/llama.cpp)
*   **Deliverables**:
    *   **Workload Database**: Save raw runs to `results/speculative_decoding_benchmark.csv`.
    *   **Theory Reference**: Update [docs/speculative-decoding.md](speculative-decoding.md) detailing native C++ speculative execution structures.
    *   **Benchmark Report**: [benchmarks/15-08-2026-speculative-decoding.md](../benchmarks/15-08-2026-speculative-decoding.md) documenting speedups and acceptance rates under dynamic hardware partitions.

---

### Month 5 — Prototype a Novel Optimization Idea

#### Weeks 17–18: Idea Journaling & Research Question Formulation
*   **Goal**: Review all performance logs to identify inefficiencies under consumer resource boundaries (e.g., dynamic $K$-selection, hybrid offloading schedules) and frame a clear research hypothesis.
*   **Reference Materials**:
    *   arXiv search listings on low-VRAM LLM serving.
*   **Deliverables**:
    *   **Learning Report**: [learnings/learning-research-proposal.md](../learnings/learning-research-proposal.md) proposing the hypothesis, target hardware configurations, and planned pipeline designs.

#### Weeks 19–20: Rapid Prototyping & Initial Evaluation
*   **Goal**: Build a minimal prototype of the proposed optimization within the local codebase and run rapid measurements to confirm if the latency/memory savings hold.
*   **Deliverables**:
    *   **Code**: Create experimental scripts under a `prototype/` directory.
    *   **Benchmark Report**: [benchmarks/15-09-2026-prototype-evaluation.md](../benchmarks/15-09-2026-prototype-evaluation.md) documenting early evaluations, execution speed deltas, and failure adjustments.

---

### Month 6 — Full Implementation, Evaluation & Technical Report

#### Weeks 21–23: Full Implementation & Broad Evaluation
*   **Goal**: Refine the prototype into a production-grade codebase, evaluate generation quality using standard harnesses, and compile a final comparison against baseline and SnapKV implementations.
*   **Reference Materials**:
    *   [EleutherAI LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness)
*   **Deliverables**:
    *   **Code**: Populate production modules inside `src/`.
    *   **Workload Database**: Save summary evaluations to [results/final_comparison.csv](../results/final_comparison.csv).
    *   **Benchmark Report**: [benchmarks/15-10-2026-final-evaluation.md](../benchmarks/15-10-2026-final-evaluation.md) compiling overall latency, throughput, and accuracy tables.

#### Week 24: Write the Technical Report
*   **Goal**: Document the entire 6-month study, the optimization mechanics, measured performance gains, trade-offs, and future directions.
*   **Deliverables**:
    *   **Learning Report**: [learnings/learning-final-technical-report.md](../learnings/learning-final-technical-report.md) serving as the comprehensive project publication and technical summary.

---

## Part 10: Benchmark Report Methodology

To ensure absolute rigor, reproducibility, and cross-comparison of findings over months of system tuning, all benchmarking runs in the workspace must be codified into a standardized, database-linked Benchmark Report.

### Standard Benchmark Report Template

```markdown
# Experiment Title

## Objective

## Research Question

## Hardware Configuration

## Software Configuration

## Model Information

## Benchmark Configuration

## Raw Results

## Statistical Summary

## Analysis

## Observations

## Unexpected Findings

## Performance Bottlenecks

## Conclusions

## Future Experiments

## Related Learnings

## Related Theory Documents
```

### Report Section Guide & Best Practices

#### 1. Experiment Title
*   **What belongs there**: A search-friendly, descriptive title encoding the model, the core independent variables under test, and the physical hardware.
*   **Why it exists**: Serves as the primary identifier in the research archive index.
*   **Examples**: `[RTX 3050 Laptop] Llama-3-8B-Q4_K_M GPU Layer Offloading Scaling Study`.
*   **Common mistakes**: Stating generic project names like "Warmup Test" or "Llama Speed".

#### 2. Objective
*   **What belongs there**: The underlying engineering motivation of the study. Explain what system property you are evaluating and why it is being targeted.
*   **Why it exists**: Frames the engineering context and prevents objective creep.
*   **Examples**: "Isolate and quantify the throughput bottleneck of system memory transfers during split CPU-GPU layer offloading of an 8B model."
*   **Common mistakes**: Vague statements like "Measure performance".

#### 3. Research Question
*   **What belongs there**: A specific, falsifiable question stating the target threshold of performance improvement or degradation.
*   **Why it exists**: Sets a quantitative pass/fail benchmark for the experiment.
*   **Examples**: "Does offloading 24 layers of Llama-3-8B to the RTX 3050 Laptop GPU yield at least a 3x increase in generation TPS compared to 12 layers?"
*   **Common mistakes**: Broad questions like "How does offloading affect token speed?".

#### 4. Hardware Configuration
*   **What belongs there**: Complete physical system profiles: exact CPU model, active host RAM channels, clock speeds, GPU VRAM capacity, TGP (Total Graphics Power) limit, and active cooling configurations.
*   **Why it exists**: Hardware capacity determines the execution roofline limits.
*   **Examples**: "Intel Core i5-12500H, 16 GB DDR5 Single-Channel RAM (4800 MHz, 38.4 GB/s peak), NVIDIA RTX 3050 Laptop (6 GB GDDR6 VRAM, 60W TGP, PCIe Gen4 x8 interface)."
*   **Common mistakes**: Writing general specs like "Laptop i5, 16GB RAM, 3050".

#### 5. Software Configuration
*   **What belongs there**: Complete software versions: OS build number, display driver versions, CUDA toolkit version, active python libraries, runtime executable build hashes, and compiler flags.
*   **Why it exists**: Differences in software compilation and driver versions introduce significant performance delta.
*   **Examples**: "Windows 11 Home 23H2 (Build 22631.3527), NVIDIA Game Ready Driver 555.99, CUDA Toolkit 12.4, `llama-cpp-python v0.2.76` compiled with `LLAMA_CUDA=ON`."
*   **Common mistakes**: Omitting library compile variables (e.g., omitting whether CUDA backend was active).

#### 6. Model Information
*   **What belongs there**: Exact model identifier, parameter size, GGUF file size, architecture type, and specific quantization method.
*   **Why it exists**: Model parameters govern weight streaming thresholds.
*   **Examples**: `Meta-Llama-3-8B-Instruct-GGUF (Q4_K_M, 8.03B parameters, 32 transformer layers, 4.80 GB footprint)`.
*   **Common mistakes**: Writing "Llama 8B" without the specific quantization suffix.

#### 7. Benchmark Configuration
*   **What belongs there**: Input prompt length, output token limit, temperature, target threads, batch size, and active caches.
*   **Why it exists**: Governs the computational workload and isolates either the prefill or decode phase.
*   **Examples**: `Prompt length = 512 tokens, generation limit = 128 tokens, temp = 0.0, threads = 6, ngl = 0 to 32 (step 4), cache = True`.
*   **Common mistakes**: Modifying values during active testing sweeps without documenting.

#### 8. Raw Results
*   **What belongs there**: Factual tabular representations of the raw outputs for every iteration.
*   **Why it exists**: Validates the dataset, ensuring transparency and auditability.
*   **Examples**: Structured Markdown tables mapping runs to specific performance values.
*   **Common mistakes**: Pasting verbose console output dumps.

#### 9. Statistical Summary
*   **What belongs there**: Aggregated metrics across all non-warmup iterations: Mean, Median, Standard Deviation, p95/p99 tail latency, and 95% Confidence Intervals.
*   **Why it exists**: Controls for noise and validates the mathematical significance of performance improvements.
*   **Examples**: Mean latency calculation accompanied by the margins of error.
*   **Common mistakes**: Averaging results without discarding the initial warm-up iterations.

#### 10. Analysis
*   **What belongs there**: Causal deduction linking the observed data back to physical hardware limits and software architectures.
*   **Why it exists**: Interprets the raw observations and answers "how the system worked".
*   **Examples**: "The throughput scaling profile remains linear until ngl=32 because the CPU execution path forces intermediate activations to step over the PCIe bus, saturating Host-to-Device copying channels."
*   **Common mistakes**: Repeating the results section in prose format without explaining the engineering cause.

#### 11. Observations
*   **What belongs there**: Factual, non-theoretical summaries of the trends visible in the data charts.
*   **Why it exists**: Establishes the undisputed baseline facts of the run.
*   **Examples**: "Throughput scaled linearly by 0.46 TPS per layer from ngl=0 to ngl=24, then jumped by 2.9x from ngl=24 to ngl=32."
*   **Common mistakes**: Mixing causal explanations directly with the description of observations.

#### 12. Unexpected Findings
*   **What belongs there**: Anomalies, outliers, or performance drops that did not align with initial predictions.
*   **Why it exists**: Serves as the primary catalyst for deep-dive root-cause investigations.
*   **Examples**: "GPU memory allocations remained high even after execution terminated, indicating a VRAM leak in the thread clean-up routines."
*   **Common mistakes**: Deleting outlier runs to make the data curve look smooth.

#### 13. Performance Bottlenecks
*   **What belongs there**: Identification of the limiting execution step (e.g., CPU thread synchronization, host memory bus bandwidth, PCIe link latency, GPU VRAM transfer speed).
*   **Why it exists**: Guides target focus for subsequent optimization steps.
*   **Examples**: "The generation speed was capped by single-channel DDR5 bus speeds (38.4 GB/s) streaming weights to the CPU execution core."
*   **Common mistakes**: Writing generic statements like "The GPU was slow".

#### 14. Conclusions
*   **What belongs there**: Factual answers to the research questions based on the collected evidence.
*   **Why it exists**: Synthesizes the results of the experiment.
*   **Examples**: "Full GPU offload is required to achieve real-time text delivery. Partial offload is viable for low-concurrency pipelines but fails interactive SLAs."
*   **Common mistakes**: Over-generalizing conclusions to unrelated hardware setups.

#### 15. Future Experiments
*   **What belongs there**: Clear, actionable next steps designed to explore new hypotheses.
*   **Why it exists**: Maintains the sequential nature of the optimization roadmap.
*   **Examples**: "Benchmark the impact of GGUF quantization levels (Q2_K through Q8_0) at full GPU offloading (`ngl = 32`) to identify the perplexity-to-throughput knee."
*   **Common mistakes**: Suggesting non-specific next steps.

#### 16. Related Learnings
*   **What belongs there**: Clickable file links pointing to deep-dive Learning Reports.
*   **Why it exists**: Bridges data observations to technical analyses.
*   **Common mistakes**: Failing to update this section as new learning files are created.

#### 17. Related Theory Documents
*   **What belongs there**: Clickable file links pointing to documentation within the local theory knowledge base.
*   **Why it exists**: Grounds the experiment within established computational model principles.
*   **Common mistakes**: Hardcoding theory definitions inside the report.

### System Interlinking Requirements
To build a cohesive knowledge repository, reports must link to external database entities:
*   **Tabular Database (`results/benchmark_history.csv`)**: Every benchmark report must refer to the exact row indices or run IDs recorded in the central CSV database.
*   **Telemetry Dump (`results/json/*.json`)**: Reference the specific timestamped JSON files containing raw hardware utilization data.
*   **Learning Documents (`learnings/*.md`)**: If a run identifies a unique systems phenomenon, link to the corresponding `learnings/learning-*.md` file.
*   **Theory Reference (`docs/*.md`)**: Link to the underlying theory files in the `docs` folder to provide architectural context.

---

### Example Benchmark Report

# [RTX 3050 Laptop] Llama-3-8B-Q4_K_M GPU Layer Offloading Scaling Study

## Objective
Establish the throughput-to-latency scaling curve when transitioning from full CPU execution to full GPU offloading under CPU-GPU split execution using llama.cpp.

## Research Question
Does offloading all 32 transformer layers of `Meta-Llama-3-8B-Instruct-GGUF` to the RTX 3050 Laptop GPU yield at least a 3x throughput improvement (TPS) compared to a partial offload of 16 layers?

## Hardware Configuration
*   **CPU**: Intel Core i5-12500H (12 Cores, 16 Threads)
*   **RAM**: 16 GB DDR5 Single-Channel (4800 MHz, 38.4 GB/s theoretical bandwidth)
*   **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB GDDR6 VRAM, 60W TGP, 192 GB/s memory bandwidth)
*   **PCIe**: PCIe Gen4 x8 interface (15.75 GB/s limit)
*   **Thermal state**: Passive cool down to <45°C GPU core temperature before each run.

## Software Configuration
*   **OS**: Windows 11 Home (23H2, Build 22631.3527)
*   **Driver**: NVIDIA Game Ready Driver 555.99
*   **Inference Engine**: `llama-cpp-python v0.2.76` (compiled with CUDA support, CUDA Toolkit 12.4)

## Model Information
*   **Name**: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
*   **Parameter Count**: 8.03B parameters
*   **Layer Count**: 32 Transformer layers
*   **File Size**: 4.80 GB

## Benchmark Configuration
*   **Prompt**: "Translate the following passage into French: [512 dummy tokens]"
*   **Prompt Tokens**: 512
*   **Generation Tokens**: 128
*   **Temperature**: 0.0 (greedy decoding)
*   **Threads**: 6 physical cores
*   **Independent Variable**: `n_gpu_layers` (ngl) evaluated at: 0, 8, 16, 24, 32.
*   **Iterations**: 10 runs per step (following 3 discarded warm-ups)

## Raw Results

| Run # | ngl = 0 (TPS) | ngl = 8 (TPS) | ngl = 16 (TPS) | ngl = 24 (TPS) | ngl = 32 (TPS) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 3.12 | 5.42 | 9.11 | 14.22 | 42.45 |
| 2 | 3.10 | 5.39 | 9.08 | 14.15 | 42.31 |
| 3 | 3.15 | 5.45 | 9.15 | 14.30 | 42.48 |
| 4 | 3.09 | 5.38 | 9.05 | 14.10 | 42.10 |
| 5 | 3.12 | 5.41 | 9.12 | 14.25 | 42.38 |
| 6 | 3.11 | 5.40 | 9.09 | 14.18 | 42.42 |
| 7 | 3.13 | 5.43 | 9.14 | 14.28 | 42.51 |
| 8 | 3.08 | 5.37 | 9.03 | 14.08 | 41.95 |
| 9 | 3.12 | 5.41 | 9.10 | 14.21 | 42.40 |
| 10 | 3.10 | 5.39 | 9.07 | 14.16 | 42.36 |

## Statistical Summary

| Metric (TPS) | ngl = 0 | ngl = 8 | ngl = 16 | ngl = 24 | ngl = 32 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean** | 3.11 | 5.41 | 9.09 | 14.19 | 42.34 |
| **Median** | 3.12 | 5.41 | 9.09 | 14.20 | 42.39 |
| **StDev** | 0.02 | 0.02 | 0.04 | 0.07 | 0.17 |
| **p95** | 3.14 | 5.44 | 9.14 | 14.29 | 42.49 |
| **95% CI** | [3.10, 3.12] | [5.40, 5.42] | [9.07, 9.11] | [14.15, 14.23] | [42.23, 42.45] |

## Analysis
The system performance scales non-linearly with GPU layer offloading:

```
TPS
 50 |                                                  * (ngl=32)
 40 |
 30 |
 20 |
 10 |                            * (ngl=24)
  0 +------*----------*----------+----------------------
        ngl=0      ngl=16      ngl=32
```

At `ngl = 16`, the model generates at a mean rate of 9.09 TPS. At `ngl = 32` (full GPU offload), the throughput jumps to 42.34 TPS—a **4.65x increase**. This validates our research question (exceeding the target 3x boundary).
The performance profile shows that partial offloading suffers from heavy host memory bandwidth limits and PCIe overhead. When `ngl < 32`, intermediate activation matrices must be transferred back and forth across the PCIe Gen4 x8 bus (with an observed throughput of ~15.75 GB/s). When `ngl = 32`, all model parameters reside in GDDR6 VRAM, and the entire computational graph is compiled and run on the GPU, eliminating the PCIe and host memory bus bottlenecks completely.

## Observations
*   Scaling from 0 to 24 offloaded layers shows a linear throughput increase of ~0.46 TPS per layer.
*   Transitioning from 24 layers to 32 layers yields a non-linear jump from 14.19 TPS to 42.34 TPS, indicating a major system bottleneck was removed.

## Unexpected Findings
During initial warm-up runs (runs 1-3, discarded), throughput was observed at 35.2 TPS for `ngl = 32`, showing a ~20% latency penalty due to the initial GPU memory allocation and CUDA kernel JIT compiling.

## Performance Bottlenecks
*   `ngl < 32`: The primary bottleneck is system RAM bandwidth (DDR5 Single-Channel) fetching intermediate activations and CPU processing overhead.
*   `ngl = 32`: The bottleneck transitions to GPU memory bandwidth (GDDR6 192 GB/s) streaming weights from VRAM to SRAM registers.

## Conclusions
Offloading all layers of Llama-3-8B Q4_K_M to the RTX 3050 Laptop is required to achieve interactive speed (>40 TPS). Partial offloading is usable but bottlenecked by CPU thread scheduling and PCIe bus transfers.

## Future Experiments
Evaluate the performance impact of GGUF quantization variants (Q2_K, Q3_K_L, Q4_K_M, and Q5_K_M) at full GPU offloading (`ngl = 32`) to find the optimal point between accuracy (perplexity) and throughput.

## Related Learnings
*   [learnings/learning-pcie-bottleneck.md](../learnings/learning-pcie-bottleneck.md)
*   [learnings/learning-single-vs-dual-channel-cpu.md](../learnings/learning-single-vs-dual-channel-cpu.md)

## Related Theory Documents
*   [docs/gpu-memory-bandwidth.md](gpu-memory-bandwidth.md)
*   [docs/quantization.md](quantization.md)

---

## Part 11: Learning Report Methodology

While a Benchmark Report answers **"What happened?"** (presenting data and statistical metrics), a Learning Report answers **"Why did it happen?"**. It captures systems-level insights, root-cause analyses, and structural findings that apply beyond a single experiment.

### Standard Learning Report Template

```markdown
# Learning Title

## Related Benchmark

## Key Observation

## Root Cause Analysis

## Theory Behind The Observation

## New Concepts Learned

## Relationships Discovered

## Questions Generated

## Future Investigation

## References
```

### Section Guide & Best Practices

#### 1. Learning Title
*   **Purpose**: State the high-level system concept or mechanics isolated by the study.
*   **Best Practices**: Keep the title focused on the technical mechanism rather than experiment names.
*   **Examples**: "Symmetric Quantization Loss in High-Weight Outlier Matrices", "WDDM Command Queue Overhead limits Under Windows-CUDA Subsystems".

#### 2. Related Benchmark
*   **Purpose**: Clickable file link pointing back to the source experiment that triggered this learning.
*   **Best Practices**: Explicitly reference the CSV Run ID and git commit hashes for traceability.

#### 3. Key Observation
*   **Purpose**: Describe the specific performance delta or anomaly that was measured.
*   **Best Practices**: Include raw values and note if they deviate from theoretical models.
*   **Examples**: "We observed a 30% speed drop when thread counts exceeded physical cores by just 1."

#### 4. Root Cause Analysis
*   **Purpose**: Explain the micro-architectural or software logic pathway causing the observation.
*   **Best Practices**: Ground the explanation in execution steps: registers, thread blocks, bus states, or scheduling policies.

#### 5. Theory Behind The Observation
*   **Purpose**: Ground the root cause in established computer science or hardware concepts.
*   **Best Practices**: Use equations, rooflines, and hardware specs to show that the system behaves logically under fundamental physical limits.

#### 6. New Concepts Learned
*   **Purpose**: Document definitions for terms or subsystems encountered during the investigation.
*   **Best Practices**: Explain the concept clearly in relation to the overall system.
*   **Examples**: Thread Affinity, Shared System Memory paging, L2 Cache line eviction, Kernel Fusion.

#### 7. Relationships Discovered
*   **Purpose**: Define formal mathematical heuristics or proportional rules discovered.
*   **Best Practices**: Keep them short, clear, and testable.
*   **Examples**: $\text{ITL Jitter} \propto \frac{\text{Thread Concurrency}}{\text{Physical Core Count}}$.

#### 8. Future Investigation
*   **Purpose**: Propose follow-up experiments to resolve new unknowns.
*   **Best Practices**: Design testable steps with specific configurations.

#### 9. References
*   **Purpose**: Provide links or citations to specs, papers, or code structures.

---

### Worked Example: Qwen2.5-7B offloading studies on RTX 3050

# Learning: The Non-Linear Performance Threshold of Full GPU Offload

## Related Benchmark
*   [benchmarks/bm-qwen2.5-7b-offload-scaling.md](../benchmarks/bm-qwen2.5-7b-offload-scaling.md)
*   CSV Entry: `RunID: qwen2.5-7b-opt-run-102`

## Key Observation
During layer-offload testing of Qwen2.5-7B-Instruct (Q4_K_M GGUF format, 28 transformer blocks) on an RTX 3050 Laptop GPU, we measured the following decoding speeds:
*   `-ngl 10` (10 layers on GPU, 18 on CPU): **6.2 TPS**
*   `-ngl 20` (20 layers on GPU, 8 on CPU): **11.8 TPS**
*   `-ngl 28` (All 28 transformer layers on GPU, output/embedding on CPU): **15.2 TPS**
*   `-ngl 33` (Full GPU offload: all layers, embeddings, and heads on GPU): **44.8 TPS**

The transition from `-ngl 28` to `-ngl 33` yielded a **2.94x speedup**, which is disproportionate to the minor difference of 5 computational layers offloaded.

```
TPS
 50 |                                                   * (ngl=33)
 40 |
 30 |
 20 |                                     * (ngl=28)
 10 |                      * (ngl=20)
  0 +------*---------------+--------------+--------------+
        ngl=10          ngl=20         ngl=28         ngl=33
```

## Root Cause Analysis
The root cause is the elimination of the Host-to-Device (H2D) and Device-to-Host (D2H) memory synchronization boundary.
When running with `-ngl 28`, the model's 28 transformer layers run on the GPU. However, the input embedding layer and the final output head (logits projection) are still processed on the CPU.
This split forces the engine to execute the following pipeline during each single token generation step:
1.  **CPU**: Token ID lookup in embedding matrix $\rightarrow$ generates activation tensor.
2.  **PCIe Transfer**: Activation tensor copied from Host RAM to GPU VRAM (H2D).
3.  **GPU**: 28 Transformer blocks process the activation tensor in VRAM.
4.  **PCIe Transfer**: Output hidden state tensor copied from VRAM back to CPU Host RAM (D2H).
5.  **CPU**: Output head computes logits $\rightarrow$ samples next token.

This introduces two PCIe copy roundtrips and two thread-scheduling synchronizations per token. Because the RTX 3050 Laptop is connected via a limited PCIe Gen4 x8 bus (with high transactional latency overhead under Windows WDDM driver virtualization), this roundtrip overhead consumes ~45 ms per step, capping generation at ~15 TPS.
When `-ngl 33` is set, `llama.cpp` offloads the embedding and output head layers to the GPU. The entire loop executes directly inside GPU VRAM, reducing PCIe memory transfers to zero and eliminating the thread synchronization blocks.

## Theory Behind The Observation
This behavior is explained by **Amdahl's Law** applied to heterogeneous computing:

$$S = \frac{1}{(1 - P) + \frac{P}{s}}$$

Where:
*   $S$ = Overall system speedup.
*   $P$ = Portion of the workload accelerated (offloaded to GPU).
*   $s$ = Speedup of the accelerated portion.

Even if the GPU accelerates the transformer blocks by 100x ($s = 100$), the remaining non-offloaded 5% of the workload running on the CPU ($1 - P = 0.05$) limits the maximum theoretical speedup to $1 / 0.05 = 20\text{x}$. In our system, the latency is dominated not by CPU math, but by the PCIe bus transfer latency:

$$\text{Latency}_{\text{step}} = \text{Latency}_{\text{compute\_gpu}} + \text{Latency}_{\text{compute\_cpu}} + \text{Latency}_{\text{PCIe\_transfer}} + \text{Latency}_{\text{driver\_overhead}}$$

When `ngl = 33`, $\text{Latency}_{\text{PCIe\_transfer}}$ and $\text{Latency}_{\text{driver\_overhead}}$ drop to zero, shifting the system from a PCIe/CPU-bound state to a GPU memory-bandwidth-bound state.

## New Concepts Learned
*   **WDDM Driver Overhead**: The Windows Display Driver Model introduces a ~1-2 ms command queuing delay for every CUDA driver submission, which severely limits high-frequency small-packet transfers.
*   **Unified Memory Fallback**: On Windows, when VRAM is exceeded, the driver silently falls back to system RAM over PCIe (shared system memory), slowing down execution without throwing an explicit OOM error.

## Relationships Discovered
$$\text{PCIe Synchronization Penalty} \propto \frac{\text{Number of Active Execution Splits}}{\text{PCIe Bandwidth}}$$

## Questions Generated
1.  Can we bypass the Windows WDDM overhead by running llama.cpp under WSL2 (Windows Subsystem for Linux), which uses a direct kernel bypass interface?
2.  What is the exact latency penalty of Windows Shared System Memory compared to active CPU layer offloading?

## Future Investigation
*   Run the same Qwen2.5-7B benchmark in WSL2 using CUDA Toolkit 12.4 and compare the partial offload TPS scaling to native Windows.
*   Monitor GPU Shared Memory allocation rates during out-of-VRAM states using NVML hooks.

## References
*   Amdahl, G. (1967). *Validity of the single processor approach to achieving large scale computing capabilities*. AFIPS Conference Proceedings.
*   NVIDIA CUDA C++ Programming Guide (v12.4), Section on Host-Device Synchronization.

---

## Part 12: Theory Knowledge Base

To prevent duplication of general systems concepts across various Benchmark and Learning Reports, the repository maintains a centralized Theory Knowledge Base under the `docs/` directory.

### Core Architecture Files

1.  **[transformer-basics.md](transformer-basics.md)**:
    *   *Purpose*: Explains Multi-Head Attention mechanics, layer architectures, and embeddings.
    *   *Scope*: Transformer structure down to vector math.
    *   *When to create/update*: When changing baseline network definitions.
    *   *References*: Reference when profiling parameter footprints or attention calculation steps.

2.  **[gpu-memory-bandwidth.md](gpu-memory-bandwidth.md)**:
    *   *Purpose*: Details arithmetic intensity limits and memory bus rooflines.
    *   *Scope*: HBM/GDDR memory configurations, bus widths, and clock frequency limits.
    *   *When to create/update*: When analyzing decode constraints or comparing system limits.
    *   *References*: Reference when discussing tokens/sec barriers on specific cards.

3.  **[kv-cache.md](kv-cache.md)**:
    *   *Purpose*: Details KV cache allocation memory sizes and architectures.
    *   *Scope*: Sizing equations for Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Grouped-Query Attention (GQA).
    *   *When to create/update*: When documenting context scaling limits.
    *   *References*: Reference when diagnosing VRAM-related Out-Of-Memory (OOM) crashes.

4.  **[quantization.md](quantization.md)**:
    *   *Purpose*: Explains mathematical precision reduction algorithms.
    *   *Scope*: Linear quantization math, scaling factors, and specific formats (AWQ, GPTQ, GGUF).
    *   *When to create/update*: When testing model throughput vs perplexity trade-offs.
    *   *References*: Reference when comparing different quantization types.

5.  **[flash-attention.md](flash-attention.md)**:
    *   *Purpose*: Documents SRAM tiling to optimize attention operations.
    *   *Scope*: Fused CUDA kernels, online softmax scaling, and write-back reduction.
    *   *When to create/update*: When analyzing prompt prefill times (TTFT) or long context windows.
    *   *References*: Reference when measuring attention computation times.

6.  **[speculative-decoding.md](speculative-decoding.md)**:
    *   *Purpose*: Details target verification scaling using lightweight draft models.
    *   *Scope*: Speculative drafting probability matrices and verification algorithms.
    *   *When to create/update*: When running multi-model decode acceleration configurations.
    *   *References*: Reference when measuring draft acceptance rates.

7.  **[continuous-batching.md](continuous-batching.md)**:
    *   *Purpose*: Explains dynamic iteration-level batch serving schedules.
    *   *Scope*: In-flight scheduling algorithms and padding elimination schemes.
    *   *When to create/update*: When testing concurrent API request scales.
    *   *References*: Reference when analyzing throughput vs. concurrency profiles.

8.  **[paged-attention.md](paged-attention.md)**:
    *   *Concept*: Virtual page translation mapping of KV cache blocks.
    *   *Scope*: Logical block structures and fragmentation elimination.
    *   *When to create/update*: When evaluating framework memory allocation pools.
    *   *References*: Reference when discussing memory fragmentation or batch allocation constraints.

9.  **[tensor-parallelism.md](tensor-parallelism.md)**:
    *   *Purpose*: Details intra-node model weight distribution.
    *   *Scope*: Row-parallel and column-parallel linear projection partitions and multi-GPU communication.
    *   *When to create/update*: When scaling benchmarks across multiple local or cloud GPUs.
    *   *References*: Reference when evaluating parallel communication overheads.

10. **[pipeline-parallelism.md](pipeline-parallelism.md)**:
    *   *Purpose*: Documents layer partitioning across multiple server nodes.
    *   *Scope*: Pipeline scheduling models (e.g., 1F1B) and communication synchronization steps.
    *   *When to create/update*: When running large model distributed benchmarks.
    *   *References*: Reference when analyzing pipeline bubbles or inter-node transfer delays.

11. **[inference-serving.md](inference-serving.md)**:
    *   *Purpose*: Documents production server routing and API gateways.
    *   *Scope*: Reverse proxy setups, queues, and SLA performance target distributions.
    *   *When to create/update*: When transitioning benchmarks from local scripts to server endpoints.
    *   *References*: Reference when comparing request latency distributions.

12. **[cuda-fundamentals.md](cuda-fundamentals.md)**:
    *   *Purpose*: Explains core GPU software execution structures.
    *   *Scope*: Threads, warps, thread blocks, shared memory allocation, and grid configurations.
    *   *When to create/update*: When writing or debugging custom CUDA/Triton kernels.
    *   *References*: Reference when profiling kernel latency metrics.

13. **[gpu-architecture.md](gpu-architecture.md)**:
    *   *Purpose*: Documents GPU hardware components.
    *   *Scope*: Streaming Multiprocessors, Tensor Cores, memory bus layout, and register files.
    *   *When to create/update*: When compiling profiles for specific hardware families (Hopper, Ada Lovelace, Ampere).
    *   *References*: Reference when discussing hardware limits (such as SM occupancy).

14. **[benchmarking-methodology.md](benchmarking-methodology.md)**:
    *   *Purpose*: Establishes standard benchmarking protocols.
    *   *Scope*: Control of variables, thermal limits, statistical significance, and margin of error math.
    *   *When to create/update*: When designing new testing frameworks.
    *   *References*: Reference when documenting the setup of experimental sweeps.

---

## Part 13: GPU Memory Bandwidth

During the token generation (decode) phase of an LLM, system performance is limited by **GPU Memory Bandwidth** rather than compute capacity.

### Why LLM Decode Is Memory-Bound
Autoregressive decoding generates text one token at a time. To output a single new token:
1.  The model's weights (often gigabytes in size) must be read from slow High Bandwidth Memory (VRAM) into the GPU core register space (SRAM).
2.  The GPU performs a matrix-vector multiplication (GEMV) using the single query vector of the active token against the model's weight tensors.
3.  Because each parameter loaded from VRAM is only used in a single operation, the **Arithmetic Intensity** of this process is extremely low:

$$\text{Arithmetic Intensity} = \frac{\text{Operations}}{\text{Bytes Transferred}} = \frac{2 \times W}{W \times P} = \frac{2}{P} \text{ FLOPs/Byte}$$

Where:
*   $W$ = Number of model parameters.
*   $P$ = Bytes per parameter (e.g., 2 bytes for FP16, 0.5 bytes for INT4).

For FP16, the arithmetic intensity is just $1 \text{ FLOP/Byte}$. If a GPU has a memory bandwidth of $192 \text{ GB/s}$, it can only load enough weights to compute $192 \text{ GFLOPs/sec}$ of math, regardless of whether the GPU is capable of 10 or 100 TFLOPs. As a result, the GPU's compute cores spend most of their time idle, waiting for weights to arrive over the memory bus.

### Why Prefill Is Compute-Bound
In contrast to the decode phase, the prompt prefill phase processes the entire sequence of input tokens simultaneously. The self-attention projection requires computing matrix-matrix multiplications (GEMM) rather than matrix-vector ones. 
If the prompt sequence length is $S$ and the batch size is $B$, the query tensor has dimensions $[B \times S \times D]$, where $D$ is the model dimension. When multiplying the weights of size $[D \times D]$ by the input tokens, each weight loaded from VRAM is reused across all $B \times S$ tokens. 
The arithmetic intensity scales linearly with sequence length:

$$\text{Arithmetic Intensity}_{\text{prefill}} \approx \frac{2 \times B \times S \times D^2}{B \times S \times D + D^2} \approx D \text{ FLOPs/Byte (for large } S\text{)}$$

With sequences of thousands of tokens, the arithmetic intensity exceeds $1000 \text{ FLOPs/Byte}$, moving the workload past the Roofline knee into the compute-bound region. The speed is limited by the GPU's Tensor Core capabilities (TFLOPs), not memory bandwidth.

### Weight Streaming Layout
The following diagram illustrates the physical weight streaming process from GPU VRAM across the internal memory bus to the SM SRAM registers during the decode phase:

```
  +-------------------------------------------------------------+
  |                        GPU VRAM (HBM/GDDR)                  |
  +-------------------------------------------------------------+
                                 |
                                 |  [Model Weights Streamed]
                                 |  (192 GB/s to 3,350 GB/s)
                                 v
  +-------------------------------------------------------------+
  |                    Memory Bus (e.g., 128-bit)               |
  +-------------------------------------------------------------+
                                 |
                                 v
  +-------------------------------------------------------------+
  |                      L1 / L2 Cache (SRAM)                   |
  +-------------------------------------------------------------+
                                 |
                                 v
  +-------------------------------------------------------------+
  |                 SM Registers / SRAM Memory                  |
  |             (Matrix Multiply / Math Engine)                 |
  +-------------------------------------------------------------+
```

### Why Quantization and Software Optimizations Help
*   **Quantization**: Reducing weight precision from FP16 (2 bytes) to INT4 (0.5 bytes) shrinks the model size by 4x. This directly reduces the memory transfer overhead, resulting in up to a 4x increase in decoding speed.
*   **FlashAttention**: Reduces VRAM access overheads by loading segments of the Q, K, and V matrices into fast SRAM, computing attention locally, and writing the output back, minimizing slow VRAM read/write cycles.
*   **TensorRT-LLM**: Implements custom fused GEMM kernels that keep weights quantized in VRAM, transferring smaller byte footprints across the bus and unpacking them to floating-point format only once they reach the GPU registers.

### Hardware Comparison Profile

| GPU Model | VRAM Capacity | Memory Bus Width | Memory Bandwidth (GB/s) | Theoretical Peak FP16 Speed (TFLOPs) | Llama-3-8B Q4 (4.5 GB) Speed Limit (Tokens/s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RTX 3050 Laptop** | 6 GB | 128-bit | ~192 GB/s | ~9.0 TFLOPs | **~42.6 tokens/sec** |
| **RTX 4060 Laptop** | 8 GB | 128-bit | ~256 GB/s | ~12.1 TFLOPs | **~56.8 tokens/sec** |
| **RTX 4090 Desktop**| 24 GB | 384-bit | ~1,008 GB/s | ~82.6 TFLOPs | **~224.0 tokens/sec** |
| **NVIDIA A100** | 80 GB | 5120-bit | ~2,039 GB/s | ~312.0 TFLOPs | **~453.1 tokens/sec** |
| **NVIDIA H100** | 80 GB | 5120-bit | ~3,350 GB/s | ~989.0 TFLOPs | **~744.4 tokens/sec** |

### Practical Implications: Qwen2.5-7B on RTX 3050 Laptop
Let's apply these hardware physics to `Qwen2.5-7B` in `Q4_K_M` GGUF quantization:
*   **Model Size**: $4.3 \text{ GB}$ (static weight size in VRAM).
*   **RTX 3050 Laptop Bandwidth**: $192 \text{ GB/s}$ maximum.
*   **Theoretical Maximum Decode Throughput**:
    $$\text{Tokens/sec}_{\text{limit}} = \frac{\text{Memory Bandwidth (GB/s)}}{\text{Model Size (GB)}} = \frac{192 \text{ GB/s}}{4.3 \text{ GB}} \approx 44.65 \text{ tokens/sec}$$

This calculation reveals that even if the RTX 3050 had infinite computation cores (infinite TFLOPs), it could never exceed **44.65 tokens/sec** during the decode phase of Qwen2.5-7B Q4_K_M because it is physically bound by the memory bandwidth of its 128-bit bus. Any speedup beyond this requires either reducing the model size via further quantization or moving to a GPU with a wider memory bus.

---

## Part 14: Research Repository Architecture

To maintain experimental history and build a structured knowledge base, the repository is organized as follows:

### Directory Structure

```text
llm-inference-lab/
│
├── README.md                 # Project introduction, hardware profiles, and setup guides
│
├── benchmark_lab/            # Interactive local playgrounds and prototype scripts
│
├── benchmarks/               # Completed Benchmark Reports (following Part 10 template)
│   └── templates/            # Standard Markdown templates for copy-pasting
│
├── learnings/                # Completed Learning Reports (following Part 11 template)
│
├── docs/                     # Central engineering roadmaps and the Theory Knowledge Base
│   ├── templates/            # Documentation standards and templates
│   └── *.md                  # Theory documents (e.g., gpu-memory-bandwidth.md)
│
├── results/                  # Raw experimental output storage
│   ├── benchmark_history.csv # Tabular database tracking all benchmarking runs
│   └── json/                 # Raw JSON dumps with system telemetry (VRAM, CPU, metrics)
│
├── notebooks/                # Jupyter notebooks for statistical analysis and plotting
│
├── scripts/                  # Performance runner scripts, system monitors, and utilities
│
└── assets/                   # Media files, diagrams, and visual assets
```

### Directory Ownership & Expected Contents

#### 1. `benchmark_lab/`
*   **Purpose**: Playgrounds for debugging scripts and evaluating configurations prior to running formal, reproducible benchmarks.
*   **Ownership**: Individual researcher sandbox.
*   **Contents**: Prototype Python files, draft shell configurations, and testing prompts.

#### 2. `benchmarks/`
*   **Purpose**: Stores finalized, reproducible Benchmark Reports.
*   **Ownership**: Public project archive.
*   **Contents**: Markdown files matching the Part 10 template detailing completed experiments.

#### 3. `learnings/`
*   **Purpose**: Stores deep-dive root-cause analyses (Learning Reports) that explain performance observations.
*   **Ownership**: Public system knowledge base.
*   **Contents**: Markdown files explaining why specific software configurations behaved the way they did.

#### 4. `docs/`
*   **Purpose**: Holds the central roadmap files and the Theory Knowledge Base.
*   **Ownership**: Public systems architecture team.
*   **Contents**: Conceptual files outlining hardware specifications and architectural definitions.

#### 5. `results/`
*   **Purpose**: The central database of raw performance records.
*   **Ownership**: System database.
*   **Contents**: A central `benchmark_history.csv` log and a `json/` directory filled with raw telemetry metrics.

#### 6. `notebooks/`
*   **Purpose**: Contains Jupyter Notebooks used to process, plot, and verify the datasets.
*   **Ownership**: Systems analysis team.
*   **Contents**: Statistical computation notebooks generating the plots used in Benchmark Reports.

#### 7. `scripts/`
*   **Purpose**: Houses executable tools used to trigger, collect, and clean benchmark results.
*   **Ownership**: Infrastructure team.
*   **Contents**: Bash/PowerShell runner scripts, performance hooks, and NVML telemetry trackers.

#### 8. `assets/`
*   **Purpose**: Stores media, diagrams, and visual references.
*   **Ownership**: Documentation designers.
*   **Contents**: PNG/SVG graphics, performance plots, and system architectures diagrams.

---

### Research Workflow & Knowledge Lifecycle

The repository operates on a continuous feedback loop where raw data is systematically processed into architectural knowledge:

```
  1. RUN BENCHMARK      ==> Scripts execute sweeps, gathering metrics.
         |
         v
  2. STORE RESULTS      ==> Outputs are appended toresults/benchmark_history.csv
         |                  and results/json/*.json.
         v
  3. ANALYZE RESULTS    ==> notebooks/ process raw records and generate plots.
         |
         v
  4. WRITE REPORT       ==> Create benchmarks/exp-*.md matching the template.
         |
         v
  5. ANALYZE CAUSES     ==> Document findings in learnings/learn-*.md.
         |
         v
  6. UPDATE THEORY      ==> Codify static concepts inside docs/*.md.
         |
         v
  7. GENERATE HYPOTHESIS ==> Formulate next steps based on theory updates.
         |
         v
  8. DESIGN EXPERIMENT  ==> Create new configs and return to Step 1.
```

### Scaling Mechanism & Real-World Alignment
This architecture scales to handle hundreds of experiments over 12+ months:
*   **Decoupled Database**: Telemetry is saved in raw CSV/JSON format, allowing researchers to re-process metrics using different statistical tools without losing original data.
*   **Centralized Theory**: Theory documents prevent researchers from rewriting attention or quantization mathematics in every single report, keeping report sizes manageable.
*   **Standardized Structure**: The template guarantees that all benchmark reports are easily indexable by automated search tools.
*   **Industry Alignment**: This structure mirrors methodologies used by ML systems research teams at DeepMind, NVIDIA, and Meta, ensuring that local work remains highly structured and rigorous.

