import os
import sys

# Ensure CUDA DLLs are found on Windows before importing llama_cpp
if sys.platform.startswith("win"):
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
    if os.path.exists(cuda_path):
        os.environ["PATH"] = cuda_path + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(cuda_path)
        except AttributeError:
            pass

import argparse
import time
import uuid
import statistics
from datetime import datetime

# Import utilities
from utils.system_monitor import SystemMonitor
from utils.metrics import calculate_tps, to_ms
from utils.logging import log_to_csv, save_json_artifact

FRAMEWORK_VERSION = "1.0.0"

def run_warmup(llm, iterations=3, tokens=5):
    """
    Runs quick warmup iterations to force CUDA initialization, memory caching,
    and prevent initial cold start latency from corrupting measurements.
    """
    print(f"\n[Warm-up] Running {iterations} warm-up iterations (generating {tokens} tokens each)...")
    for i in range(1, iterations + 1):
        start = time.perf_counter()
        stream = llm(
            "Warmup iteration query.",
            max_tokens=tokens,
            temperature=0.7,
            stream=True
        )
        for _ in stream:
            pass
        duration = time.perf_counter() - start
        print(f"  Warm-up {i}/{iterations} completed in {duration:.4f}s")
    print("[Warm-up] Finished successfully.\n")

def get_stats(data_list):
    """Computes mean, standard deviation, min, max, median, and 95th percentile for a list of numbers."""
    if not data_list:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "p95": 0.0}
    
    mean = statistics.mean(data_list)
    stdev = statistics.stdev(data_list) if len(data_list) > 1 else 0.0
    
    sorted_data = sorted(data_list)
    n = len(sorted_data)
    
    median = statistics.median(sorted_data)
    
    # Calculate 95th percentile
    p95_idx = max(0, min(n - 1, int(round(n * 0.95) - 1)))
    p95 = sorted_data[p95_idx]
    
    return {
        "mean": round(mean, 2),
        "stdev": round(stdev, 2),
        "min": round(min(data_list), 2),
        "max": round(max(data_list), 2),
        "median": round(median, 2),
        "p95": round(p95, 2)
    }

def main():
    parser = argparse.ArgumentParser(description="Local LLM Inference Benchmark Framework")
    parser.add_argument("--model", type=str, required=True, help="Path to the GGUF model file")
    parser.add_argument("--quant", type=str, required=True, help="Quantization format (e.g. Q4_K_M)")
    parser.add_argument("--prompt", type=str, required=True, help="Path to the prompt text file")
    parser.add_argument("--ngl", type=int, default=33, help="Number of GPU layers to offload")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of tokens to generate")
    parser.add_argument("--ctx", type=int, default=4096, help="Context length limit")
    parser.add_argument("--temp", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--rep_pen", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--iterations", type=int, default=1, help="Number of times to repeat the benchmark")
    
    args = parser.parse_args()
    
    # 1. Resolve paths
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found at: {args.model}")
    if not os.path.exists(args.prompt):
        raise FileNotFoundError(f"Prompt file not found at: {args.prompt}")
        
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
        
    # Extract model name from filename
    model_filename = os.path.basename(args.model)
    model_name = os.path.splitext(model_filename)[0]
    
    # Generate Unique Run ID
    run_dt = datetime.now()
    run_id = f"RUN-{run_dt.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    
    # Initialize system monitor
    monitor = SystemMonitor(interval=0.1)
    
    # Resolve system hardware info
    cpu_name = monitor.get_cpu_name()
    total_ram = monitor.get_total_ram_gb()
    gpu_name = monitor.get_gpu_name()
    total_vram = monitor.get_total_vram_gb()
    
    print("=" * 60)
    print("           INITIALIZING BENCHMARK RUN")
    print("=" * 60)
    print(f"Run ID: {run_id} | Framework Version: {FRAMEWORK_VERSION}")
    print("-" * 60)
    print(f"Hardware:")
    print(f"  CPU:  {cpu_name}")
    print(f"  RAM:  {total_ram} GB")
    print(f"  GPU:  {gpu_name}")
    print(f"  VRAM: {total_vram} GB")
    print("-" * 60)
    print(f"Model: {model_filename}")
    print(f"  Quantization:   {args.quant}")
    print(f"  GPU Offload:    {args.ngl} layers")
    print(f"  Context Length: {args.ctx}")
    print(f"  Iterations:     {args.iterations}")
    print("=" * 60)

    # 2. Profile memory state before loading model
    ram_before_load = monitor.get_ram_usage_mb()
    process_ram_before_load = monitor.get_process_ram_usage_mb()
    vram_before_load = monitor.get_vram_usage_mb()

    # 3. Load the model
    print("\nLoading model in llama-cpp-python...")
    from llama_cpp import Llama
    
    llm = Llama(
        model_path=args.model,
        n_gpu_layers=args.ngl,
        n_ctx=args.ctx,
        verbose=False
    )
    print("Model loaded successfully.")
    
    # 4. Profile memory state after loading model (weights memory)
    ram_after_load = monitor.get_ram_usage_mb()
    process_ram_after_load = monitor.get_process_ram_usage_mb()
    vram_after_load = monitor.get_vram_usage_mb()
    
    # Compute weight load sizes
    weight_ram_mb = max(0.0, ram_after_load - ram_before_load)
    weight_vram_mb = max(0.0, vram_after_load - vram_before_load)
    print(f"Model weights memory footprint:")
    print(f"  System RAM increase: {weight_ram_mb:.2f} MB")
    print(f"  System VRAM increase: {weight_vram_mb:.2f} MB")
    print("-" * 60)

    # 5. Warm-up
    run_warmup(llm)
    
    # Run iterations
    raw_runs = []
    
    for iter_idx in range(1, args.iterations + 1):
        print(f"\nRunning Iteration {iter_idx}/{args.iterations}...")
        
        # Reset llama.cpp internal C++ timings
        try:
            import llama_cpp
            llama_cpp.llama_perf_context_reset(llm.ctx)
        except Exception:
            pass
            
        # Measure initial memory states
        ram_before = monitor.get_ram_usage_mb()
        process_ram_before = monitor.get_process_ram_usage_mb()
        vram_before = monitor.get_vram_usage_mb()
        
        # Count prompt tokens
        prompt_tokens = len(llm.tokenize(prompt_text.encode("utf-8")))
        
        # Start resource tracking
        monitor.start()
        
        # Execute generation
        start_time = time.perf_counter()
        first_token_time = None
        generated_text = ""
        
        stream = llm(
            prompt_text,
            max_tokens=args.max_tokens,
            temperature=args.temp,
            repeat_penalty=args.rep_pen,
            stream=True
        )
        
        # If running multiple iterations, only stream outputs for the first iteration
        # to keep console clean and prevent spam.
        is_silent = iter_idx > 1
        if not is_silent:
            print("Streaming generation response:")
            print("-" * 60)
            
        for chunk in stream:
            text = chunk["choices"][0]["text"]
            if first_token_time is None:
                first_token_time = time.perf_counter()
            if text:
                generated_text += text
                if not is_silent:
                    print(text, end="", flush=True)
                    
        end_time = time.perf_counter()
        if not is_silent:
            print("\n" + "-" * 60)
            
        # Collect post-generation states
        ram_after = monitor.get_ram_usage_mb()
        process_ram_after = monitor.get_process_ram_usage_mb()
        vram_after = monitor.get_vram_usage_mb()
        monitor_stats = monitor.stop()
        
        # 6. Retrieve exact C++-level performance timings
        t_p_eval_ms = None
        t_eval_ms = None
        cpp_generated_tokens = None
        
        try:
            perf_data = llama_cpp.llama_perf_context(llm.ctx)
            t_p_eval_ms = perf_data.t_p_eval_ms
            t_eval_ms = perf_data.t_eval_ms
            cpp_generated_tokens = perf_data.n_eval
        except Exception:
            pass
            
        # Tokenize output to get exact count as fallback or verification
        generated_tokens_python = len(llm.tokenize(generated_text.encode("utf-8")))
        generated_tokens = cpp_generated_tokens if (cpp_generated_tokens and cpp_generated_tokens > 0) else generated_tokens_python
        
        # Compute Python level fallback latencies
        total_request_latency = end_time - start_time
        first_token_latency = (first_token_time - start_time) if first_token_time else total_request_latency
        total_generation_time = (end_time - first_token_time) if first_token_time else total_request_latency
        
        first_token_latency_ms = to_ms(first_token_latency)
        total_generation_time_ms = to_ms(total_generation_time)
        total_request_latency_ms = to_ms(total_request_latency)
        
        # Resolve metrics using C++ data if available
        # Note: t_p_eval_ms is prompt processing time; t_eval_ms is token generation time
        prompt_eval_time_ms = t_p_eval_ms if (t_p_eval_ms and t_p_eval_ms > 0) else first_token_latency_ms
        generation_time_ms = t_eval_ms if (t_eval_ms and t_eval_ms > 0) else total_generation_time_ms
        
        # Compute throughputs & ITL
        prompt_tps = calculate_tps(prompt_tokens, prompt_eval_time_ms)
        generation_tps = calculate_tps(generated_tokens, generation_time_ms)
        avg_itl_ms = round(generation_time_ms / generated_tokens, 2) if generated_tokens > 0 else 0.0
        
        iter_metrics = {
            "run_id": run_id,
            "iteration_index": iter_idx,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model_name,
            "quantization": args.quant,
            "model_path": args.model,
            "n_gpu_layers": args.ngl,
            "context_length": args.ctx,
            "max_tokens": args.max_tokens,
            "temperature": args.temp,
            "repetition_penalty": args.rep_pen,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "prompt_tps": prompt_tps,
            "generation_tps": generation_tps,
            "first_token_latency_ms": first_token_latency_ms,
            "total_generation_time_ms": generation_time_ms,
            "total_request_latency_ms": total_request_latency_ms,
            "avg_itl_ms": avg_itl_ms,
            "ram_before_mb": ram_before,
            "ram_after_mb": ram_after,
            "process_ram_before_mb": process_ram_before,
            "process_ram_after_mb": process_ram_after,
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after,
            "cpu_utilization": monitor_stats["cpu_utilization"],
            "gpu_utilization": monitor_stats["gpu_utilization"],
            "gpu_mem_utilization": monitor_stats["gpu_memory_utilization"],
            "gpu_temperature_c": monitor_stats["gpu_temperature_c"],
            "gpu_power_watts": monitor_stats["gpu_power_watts"],
            "gpu_graphics_clock_mhz": monitor_stats["gpu_graphics_clock_mhz"],
            "gpu_memory_clock_mhz": monitor_stats["gpu_memory_clock_mhz"],
            "ram_before_load_mb": ram_before_load,
            "ram_after_load_mb": ram_after_load,
            "vram_before_load_mb": vram_before_load,
            "vram_after_load_mb": vram_after_load
        }
        
        # Save raw CSV row for historical database
        log_to_csv("results/benchmark_history.csv", iter_metrics)
        raw_runs.append(iter_metrics)
        
        if is_silent:
            print(f"  Done (Generated {generated_tokens} tokens | TPS: {generation_tps:.2f} | ITL: {avg_itl_ms:.2f} ms)")

    monitor.close()

    # 7. Compute Consolidated Summary across all runs
    summary_metrics = ["first_token_latency_ms", "total_generation_time_ms", "total_request_latency_ms", "avg_itl_ms", "prompt_tps", "generation_tps"]
    metrics_summary = {}
    for m in summary_metrics:
        metrics_summary[m] = get_stats([r[m] for r in raw_runs])
        
    hardware_metrics = ["cpu_utilization", "gpu_utilization", "gpu_mem_utilization", "gpu_temperature_c", "gpu_power_watts", "gpu_graphics_clock_mhz", "gpu_memory_clock_mhz"]
    hardware_summary = {}
    for h in hardware_metrics:
        # Map stats for keys
        stats_data = get_stats([r[h] for r in raw_runs])
        hardware_summary[f"{h}_summary"] = {
            "mean": stats_data["mean"],
            "stdev": stats_data["stdev"]
        }
        
    # Compile JSON artifact payload
    json_payload = {
        "framework_version": FRAMEWORK_VERSION,
        "run_id": run_id,
        "timestamp": run_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": {
            "cpu": cpu_name,
            "ram_gb": total_ram,
            "gpu": gpu_name,
            "vram_gb": total_vram
        },
        "model": {
            "name": model_name,
            "quantization": args.quant,
            "model_path": args.model
        },
        "configuration": {
            "n_gpu_layers": args.ngl,
            "context_length": args.ctx,
            "max_tokens": args.max_tokens,
            "temperature": args.temp,
            "repetition_penalty": args.rep_pen,
            "iterations": args.iterations
        },
        "memory_load_summary": {
            "ram_before_load_mb": ram_before_load,
            "ram_after_load_mb": ram_after_load,
            "process_ram_before_load_mb": process_ram_before_load,
            "process_ram_after_load_mb": process_ram_after_load,
            "vram_before_load_mb": vram_before_load,
            "vram_after_load_mb": vram_after_load
        },
        "metrics_summary": metrics_summary,
        "hardware_summary": hardware_summary,
        "runs": raw_runs
    }
    
    # Save the consolidated JSON file
    json_path = save_json_artifact("results/json", json_payload, run_dt)
    
    # 8. Print clean, beautiful consolidated table summary
    t_sum = metrics_summary["generation_tps"]
    ttft_sum = metrics_summary["first_token_latency_ms"]
    itl_sum = metrics_summary["avg_itl_ms"]
    
    print("\n" + "=" * 60)
    print("             CONSOLIDATED BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Model Context:    {model_name} ({args.quant})")
    print(f"Prompt Length:    {prompt_tokens} tokens")
    print(f"Iterations:       {args.iterations} runs")
    print("-" * 60)
    print("Latencies (ms):")
    print(f"  First Token Latency (TTFT):   mean={ttft_sum['mean']:.2f} ms | stdev={ttft_sum['stdev']:.2f} ms | p95={ttft_sum['p95']:.2f} ms")
    print(f"  Inter-Token Latency (ITL):    mean={itl_sum['mean']:.2f} ms | stdev={itl_sum['stdev']:.2f} ms | p95={itl_sum['p95']:.2f} ms")
    print("-" * 60)
    print("Throughput (tokens/sec):")
    print(f"  Token Generation (Decode):    mean={t_sum['mean']:.2f} t/s | stdev={t_sum['stdev']:.2f} t/s | min={t_sum['min']:.2f} t/s | max={t_sum['max']:.2f} t/s")
    print("Utilization Averages:")
    print(f"  System RAM Before / Load:     {ram_before_load:.2f} MB / {ram_after_load:.2f} MB")
    print(f"  System VRAM Before / Load:    {vram_before_load:.2f} MB / {vram_after_load:.2f} MB")
    print(f"  Average CPU Utilization:      {hardware_summary['cpu_utilization_summary']['mean']:.2f} %")
    print(f"  Average GPU Utilization:      {hardware_summary['gpu_utilization_summary']['mean']:.2f} %")
    print(f"  Average GPU Memory Controller:{hardware_summary['gpu_mem_utilization_summary']['mean']:.2f} %")
    print(f"  Average GPU Power Draw:       {hardware_summary['gpu_power_watts_summary']['mean']:.2f} W")
    print(f"  Average GPU Temperature:      {hardware_summary['gpu_temperature_c_summary']['mean']:.2f} °C")
    print("-" * 60)
    print(f"CSV History Appended: results/benchmark_history.csv")
    print(f"JSON Artifact Created: {json_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
