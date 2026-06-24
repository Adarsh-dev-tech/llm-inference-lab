# Master Task List: LLM Inference Optimization Curriculum

This master task list tracks the execution of the 6-Month Project-First LLM Inference Optimization curriculum. Each task details step-by-step execution steps, learning reference links, deadlines, and expected file deliverables routed to their correct directories.

---

## Month 1 — Build the Benchmark Lab

### [x] Week 1–2: Set Up llama.cpp & Run Your First Model
- **Deadline**: End of Week 2
- **Step-by-Step Execution Details**:
  1. Create the project root folder `llm-inference-lab` and initialize the git repository.
  2. Set up a Python virtual environment (`.venv`) and install baseline dependencies.
  3. Compile `ggerganov/llama.cpp` from source with CUDA acceleration flag `GGML_CUDA=ON` to utilize the NVIDIA RTX 3050 Laptop GPU.
  4. Download the baseline model: `Qwen2.5-7B-Instruct` in `Q4_K_M` GGUF format from Hugging Face.
  5. Run test generations using the native `llama-cli` tool.
  6. Experiment with the offloaded layers flag (`-ngl` / `--n-gpu-layers`) from 0 to 33. Observe when CPU vs GPU memory allocations shift by keeping `nvidia-smi` and system monitors open in secondary terminals.
- **Reference & Learning Materials**:
  - [llama.cpp Repository](https://github.com/ggerganov/llama.cpp) — Build instructions for Windows CUDA compilation.
  - [Hugging Face Qwen2.5-7B-Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) — Quantized baseline weights.
  - [Tim Dettmers' GPU Guide](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) — High-level details on VRAM sizes and memory bandwidth constraints.
  - [JAX Scaling Book Roofline Model](https://jax-ml.github.io/scaling-book/roofline/) — Guide to identifying compute-bound vs memory-bound bottlenecks.
- **Expected Deliverables**:
  - **Theory Reference**: Initialize [docs/benchmarking-methodology.md](benchmarking-methodology.md) (controls, thermal locks).
  - **Benchmark Report**: [benchmarks/12-06-2026-baseline-benchmark.md](../benchmarks/12-06-2026-baseline-benchmark.md) (documenting partial vs full GPU offload baseline timings).
  - **Learning Report**: [learnings/learning-hardware-setup.md](../learnings/learning-hardware-setup.md) detailing compilation issues, `-ngl` findings, and memory-bandwidth observations.

---

### [x] Week 3–4: Build the Benchmark Script
- **Deadline**: End of Week 4
- **Step-by-Step Execution Details**:
  1. Install `llama-cpp-python` with CUDA support in the active `.venv`.
  2. Create standard text prompt files in a `prompts/` directory: `short.txt` (~128 tokens), `medium.txt` (~512 tokens), and `long.txt` (~1024 tokens) to represent realistic query lengths.
  3. Author `utils/system_monitor.py` utilizing `psutil` and `GPUtil` to measure system RAM (RSS) and GPU VRAM footprint during generation runs.
  4. Author `utils/metrics.py` to intercept token timings and calculate Time-to-First-Token (TTFT), generation tokens per second (TPS), and inter-token latency (ITL).
  5. Author `utils/logging.py` to write raw run logs to `results/json/` and append summary parameters to `results/benchmark_history.csv`.
  6. Implement the master orchestration script `benchmark.py` that processes a warmup run before conducting testing sweeps across quantization levels (`Q4_K_M`, `Q5_K_M`, `Q8_0`, `F16`) and prompt lengths.
- **Reference & Learning Materials**:
  - [llama-cpp-python API Reference](https://github.com/abetlen/llama-cpp-python) — Guide to using programmatic bindings.
  - [GGUF Specifications](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) — Understanding block-wise quantization types.
  - [3Blue1Brown Linear Algebra Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — Deep dive into vector-matrix mapping and dimensions scaling.
- **Expected Deliverables**:
  - **Code**: [benchmark.py](../benchmark.py), [utils/system_monitor.py](../utils/system_monitor.py), [utils/metrics.py](../utils/metrics.py), and [utils/logging.py](../utils/logging.py).
  - **Workload Database**: [results/benchmark_history.csv](../results/benchmark_history.csv) (persisted history) and [results/json/](../results/json) (raw run details).
  - **Theory Reference**: Initialize [docs/quantization.md](quantization.md) (precision formats theory).
  - **Benchmark Report**: [benchmarks/15-06-2026-quantization-benchmark.md](../benchmarks/15-06-2026-quantization-benchmark.md) comparing speed and resource usage across quants.
  - **Learning Report**: [learnings/learning-quantization-differences.md](../learnings/learning-quantization-differences.md) explaining why Q4_K_M is the default for a 6GB VRAM target.

---

## Month 2 — Transformer Internals & KV Cache Profiling

### [x] Week 5–6: Build a Tiny GPT
- **Deadline**: End of Week 6
- **Step-by-Step Execution Details**:
  1. Follow Karpathy's "micrograd" tutorial to build backpropagation mechanics from scratch.
  2. Implement a decoder-only GPT model in PyTorch following the "Let's build GPT" code tutorial, training it on a character-level text dataset.
  3. Trace and log intermediate tensor shapes throughout the forward pass: input tokens -> word/position embeddings -> multi-head self-attention (Q, K, V matrices and attention weights) -> MLP layers -> layer norms -> output logits.
  4. Write a toy key-value caching layer within the PyTorch attention block to dynamically return the cached states instead of recomputing past keys/values at each decoding step. Plot time elapsed per step with vs. without caching.
- **Reference & Learning Materials**:
  - [Andrej Karpathy's Neural Nets Spelled-out Intro](https://www.youtube.com/watch?v=VMj-3S1tku0) — Building backpropagation.
  - [Andrej Karpathy's GPT From Scratch Spelled-out Intro](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Building a Transformer.
  - [Jay Alammar's The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Visual representation of model components.
  - [Lilian Weng's Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — Mathematical context on the dot-product similarity metrics.
- **Expected Deliverables**:
  - **Code**: Create `tiny_gpt/model.py`, `tiny_gpt/train.py`, and `tiny_gpt/cache_comparison.py`.
  - **Theory Reference**: Initialize [docs/transformer-basics.md](transformer-basics.md) detailing multi-head attention and architecture shapes.
  - **Learning Report**: [learnings/learning-transformer-internals.md](../learnings/learning-transformer-internals.md) mapping transformer weights, dynamic caching overhead, and context limit mathematics.

---

### [x] Week 7–8: Build a KV Cache Memory Profiler
- **Deadline**: End of Week 8
- **Step-by-Step Execution Details**:
  1. Create a Python script `kv_cache_profiler.py` using `llama-cpp-python`.
  2. Measure baseline VRAM memory footprints under increasing context windows (128, 256, 512, 1024, 2048, 4096 tokens) during the prefill and decoding stages.
  3. Validate measurements against the theoretical equation: $2 \times \text{layers} \times \text{heads} \times \text{head\_dim} \times \text{context\_len} \times \text{bytes\_per\_element}$.
  4. Identify the "OOM threshold" where memory swapping (or crashes) occurs on the RTX 3050 Laptop 6GB GPU.
- **Reference & Learning Materials**:
  - [Lilian Weng's Inference Optimization Post](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) — Sizing equations for Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Grouped-Query Attention (GQA).
  - Hugging Face transformers model configuration files to extract Qwen2.5-7B dimension config ($n_{layers}=28$, $n_{heads}=28$, $d_{head}=128$).
- **Expected Deliverables**:
  - **Code**: Create [kv_cache_profiler.py](../kv_cache_profiler.py).
  - **Workload Database**: Save metrics to [results/kv_cache_growth.csv](../results/kv_cache_growth.csv).
  - **Theory Reference**: Initialize [docs/kv-cache.md](kv-cache.md) detailing sizing mathematics and allocation logic.
  - **Benchmark Report**: [benchmarks/25-06-2026-kv-cache-profiler.md](../benchmarks/25-06-2026-kv-cache-profiler.md) plotting the memory consumption profile of baseline Qwen models.
  - **Learning Report**: [learnings/learning-kv-cache-math.md](../learnings/learning-kv-cache-math.md) detailing the VRAM boundaries, allocations, and how bandwidth constraints emerge.

---

## Month 3 — Paper Reading & SnapKV Implementation

### [ ] Week 9: Learn How to Read a Paper & Study FlashAttention
- **Deadline**: End of Week 9
- **Step-by-Step Execution Details**:
  1. Study S. Keshav's paper reading guidelines to structure paper analysis into three distinct passes.
  2. Apply the guidelines to the FlashAttention paper:
     - Pass 1: Focus on abstract, title, and high-level headings.
     - Pass 2: Analyze figures, algorithms, and key equations.
     - Pass 3: Review math details, memory layout constraints, and hardware-level tiling explanations.
  3. Document why standard PyTorch attention is bound by high-bandwidth memory (HBM) IO limitations rather than compute constraints.
- **Reference & Learning Materials**:
  - [S. Keshav's "How to Read a Paper"](http://ccr.sigcomm.org/online/files/p83-keshavA.pdf) — Three-pass methodology.
  - [Aleksa Gordić's ELI5 FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) — Simple visual and text breakdown.
  - [FlashAttention Paper (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) — PDF manuscript.
- **Expected Deliverables**:
  - **Theory Reference**: Initialize [docs/flash-attention.md](flash-attention.md) outlining fused kernels and SRAM tiles.
  - **Learning Report**: [learnings/learning-flashattention-memory-io.md](../learnings/learning-flashattention-memory-io.md) detailing hardware-level analyses of HBM transfers vs kernel processing.

---

### [ ] Week 10–12: Implement SnapKV and Benchmark It
- **Deadline**: End of Week 12
- **Step-by-Step Execution Details**:
  1. Read the SnapKV paper to understand clustering and retention of key-value pairs using attention weights during prefill.
  2. Implement a custom PyTorch/Transformers execution hook that intercepts and modifies the `past_key_values` attention state during text generation.
  3. Run a parameter sweep on key-value retention limit $K$ (e.g., $K \in \{16, 32, 64, 128, 256\}$).
  4. Compute generation quality using `llama.cpp`'s built-in perplexity tool on standard datasets (e.g., WikiText-2).
  5. Quantify the trade-off between memory footprint savings and output quality degradation.
- **Reference & Learning Materials**:
  - [SnapKV Paper (Li et al., 2024)](https://arxiv.org/abs/2404.14469) — Compression algorithm spec.
  - [SnapKV GitHub Repository](https://github.com/FasterDecoding/SnapKV) — Reference implementation.
- **Expected Deliverables**:
  - **Code**: Create `snapkv/hook.py` and `snapkv/eval.py`.
  - **Workload Database**: Save sweeps to `results/snapkv_benchmark.csv`.
  - **Theory Reference**: Update [docs/kv-cache.md](kv-cache.md) adding attention pooling and compression algorithms.
  - **Benchmark Report**: [benchmarks/15-07-2026-snapkv-compression.md](../benchmarks/15-07-2026-snapkv-compression.md) showing VRAM savings and quality (perplexity) parameters over sweeps.
  - **Learning Report**: [learnings/learning-snapkv-mechanics.md](../learnings/learning-snapkv-mechanics.md) covering hook insertions, key retention metrics, and consumer GPU execution efficiency.

---

## Month 4 — PagedAttention, Speculative Decoding & llama.cpp Tracing

### [ ] Week 13–14: Read PagedAttention & Speculative Decoding
- **Deadline**: End of Week 14
- **Step-by-Step Execution Details**:
  1. Study virtual memory page allocations for KV cache blocks (PagedAttention) and speculative generation architectures.
  2. Analyze the papers using the Keshav 3-pass methodology.
  3. Document the specific constraints that consumer GPUs (6GB VRAM) face under these techniques (e.g. block size latency scaling, dual-model VRAM space competition).
- **Reference & Learning Materials**:
  - [vLLM Blog Post on PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — Explaining logical blocks and page allocation grids.
  - [Speculative Decoding Paper (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192) — Core verification theorem.
  - [Lilian Weng's Speculative Decoding Guide](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) — Sizing target vs draft models.
- **Expected Deliverables**:
  - **Theory Reference**: Initialize [docs/paged-attention.md](paged-attention.md) (block mapping layouts) and [docs/speculative-decoding.md](speculative-decoding.md) (acceptance criteria/draft networks).
  - **Learning Report**: [learnings/learning-speculative-decoding-constraints.md](../learnings/learning-speculative-decoding-constraints.md) highlighting VRAM capacity limits and latency tradeoffs during dual-model serving on consumer chips.

---

### [ ] Week 15–16: Read llama.cpp Source & Measure Speculative Decoding
- **Deadline**: End of Week 16
- **Step-by-Step Execution Details**:
  1. Explore `ggerganov/llama.cpp` codebase. Trace the key-value buffer allocations and sampling loops.
  2. Configure speculative decoding using `llama.cpp` native CLI with a small draft model (e.g., Qwen2.5-1B-Instruct-GGUF) and the Qwen2.5-7B baseline.
  3. Benchmark speedups (TPS) and acceptance rates. Analyze if the draft model loading overhead degrades overall throughput on 6GB VRAM.
- **Reference & Learning Materials**:
  - [llama.cpp Source Code](https://github.com/ggerganov/llama.cpp) — Trace `llama_decode` in `src/llama.cpp` and `common/common.cpp` CLI parameters (like `-md` draft paths).
  - [Hugging Face Qwen2.5-1B-Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-1B-Instruct-GGUF) — Light draft weights.
- **Expected Deliverables**:
  - **Workload Database**: Save raw runs to `results/speculative_decoding_benchmark.csv`.
  - **Theory Reference**: Update [docs/speculative-decoding.md](speculative-decoding.md) detailing native C++ speculative execution structures.
  - **Benchmark Report**: [benchmarks/15-08-2026-speculative-decoding.md](../benchmarks/15-08-2026-speculative-decoding.md) documenting speedups and acceptance rates under dynamic hardware partitions.

---

## Month 5 — Prototype a Novel Optimization Idea

### [ ] Week 17–18: Idea Journaling & Research Question Formulation
- **Deadline**: End of Week 18
- **Step-by-Step Execution Details**:
  1. Analyze prior benchmarking datasets (VRAM logs, speed profiles, and perplexity data).
  2. Identify the most critical inefficiency under the 6GB VRAM constraint (e.g. dynamic attention block selection, mixed-precision layer allocation).
  3. Write a formal research question, hypothesis, and literature review.
- **Reference & Learning Materials**:
  - [arXiv Search Portal](https://arxiv.org/search/?query=LLM+inference+consumer+hardware&searchtype=all) — Querying literature on dynamic cache sizing, layer partitioning, or memory saving on consumer chips.
  - Local CSV histories and logs.
- **Expected Deliverables**:
  - **Learning Report**: [learnings/learning-research-proposal.md](../learnings/learning-research-proposal.md) proposing the hypothesis, target hardware configurations, and planned pipeline designs.

---

### [ ] Week 19–20: Rapid Prototyping & Initial Evaluation
- **Deadline**: End of Week 20
- **Step-by-Step Execution Details**:
  1. Build a quick Python/PyTorch prototype of the proposed optimization.
  2. Run initial performance tests. Collect raw metrics to see if the hypothesis holds.
  3. Adapt parameters quickly to debug accuracy or performance bottlenecks.
- **Reference & Learning Materials**:
  - Local codebase libraries and PyTorch hook guides.
  - Community feedback (e.g., share design notes on r/LocalLLaMA).
- **Expected Deliverables**:
  - **Code**: Create experimental scripts under a `prototype/` directory.
  - **Benchmark Report**: [benchmarks/15-09-2026-prototype-evaluation.md](../benchmarks/15-09-2026-prototype-evaluation.md) documenting early evaluations, execution speed deltas, and failure adjustments.

---

## Month 6 — Full Implementation, Evaluation & Technical Report

### [ ] Week 21–23: Full Implementation & Broad Evaluation
- **Deadline**: End of Week 23
- **Step-by-Step Execution Details**:
  1. Clean up and structure the prototype into a production-ready optimization library.
  2. Benchmark the final library against standard baselines (unmodified baseline and SnapKV) using `benchmark.py`.
  3. Evaluate generation quality using standard benchmarks from `EleutherAI/lm-evaluation-harness`.
- **Reference & Learning Materials**:
  - [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) — Standard accuracy suites.
  - Custom `benchmark.py` suite.
- **Expected Deliverables**:
  - **Code**: Populate production modules inside `src/`.
  - **Workload Database**: Save summary evaluations to [results/final_comparison.csv](../results/final_comparison.csv).
  - **Benchmark Report**: [benchmarks/15-10-2026-final-evaluation.md](../benchmarks/15-10-2026-final-evaluation.md) compiling overall latency, throughput, and accuracy tables.

---

### [ ] Week 24: Write the Technical Report
- **Deadline**: End of Week 24
- **Step-by-Step Execution Details**:
  1. Compile all benchmark results, prototype modifications, and insights.
  2. Author a comprehensive technical report explaining the optimization mechanism, quantitative speed/memory improvements, and failure modes.
- **Reference & Learning Materials**:
  - Local reports and learnings archives.
- **Expected Deliverables**:
  - **Learning Report**: [learnings/learning-final-technical-report.md](../learnings/learning-final-technical-report.md) serving as the comprehensive project publication and technical summary.
