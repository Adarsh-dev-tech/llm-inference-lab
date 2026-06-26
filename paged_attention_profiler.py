import os
import sys
import time
import argparse
import subprocess
import json
import csv
from datetime import datetime

# Ensure CUDA DLLs are found on Windows before importing llama_cpp
if sys.platform.startswith("win"):
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
    if os.path.exists(cuda_path):
        os.environ["PATH"] = cuda_path + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(cuda_path)
        except AttributeError:
            pass

# Import SystemMonitor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.system_monitor import SystemMonitor

def run_child_benchmark(model_path, ctx_len, ngl, use_cache):
    """
    Executes a single measurement run in a child subprocess.
    """
    monitor = SystemMonitor()
    ambient_vram = monitor.get_vram_usage_mb()
    
    from llama_cpp import Llama
    from llama_cpp.llama_cache import LlamaRAMCache
    
    start_load = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=ngl,
        n_ctx=ctx_len,
        verbose=False
    )
    load_time = time.perf_counter() - start_load
    post_load_vram = monitor.get_vram_usage_mb()
    
    # Configure prompt caching if requested
    if use_cache:
        # Allocate a 512 MB RAM cache for GGUF states
        cache = LlamaRAMCache(capacity_bytes=512 * 1024 * 1024)
        llm.set_cache(cache)
        
    # We use a long static prefix (e.g. 2000 tokens)
    prefix_tokens = [100] * 2000
    suffix_1 = [101] * 10
    suffix_2 = [102] * 10
    
    # Run the first query (Cold run / Cache population)
    prompt_1 = prefix_tokens + suffix_1
    
    # Reset performance timers
    try:
        import llama_cpp
        llama_cpp.llama_perf_context_reset(llm.ctx)
    except Exception:
        pass
        
    start_eval_1 = time.perf_counter()
    llm(prompt_1, max_tokens=1)
    eval_time_1 = time.perf_counter() - start_eval_1
    vram_after_eval_1 = monitor.get_vram_usage_mb()
    
    # Fetch C++ timings for prompt 1 evaluation if available
    try:
        perf_data = llama_cpp.llama_perf_context(llm.ctx)
        t_p_eval_ms_1 = perf_data.t_p_eval_ms
        if t_p_eval_ms_1 > 0:
            eval_time_1 = t_p_eval_ms_1 / 1000.0
    except Exception:
        pass
        
    # Run the second query (Warm run / Cache hit query)
    # We change the last few tokens to ensure it evaluates the suffix but hits on prefix
    prompt_2 = prefix_tokens + suffix_2
    
    # Reset performance timers again
    try:
        llama_cpp.llama_perf_context_reset(llm.ctx)
    except Exception:
        pass
        
    if not use_cache:
        llm.reset()
        
    start_eval_2 = time.perf_counter()
    llm(prompt_2, max_tokens=1)
    eval_time_2 = time.perf_counter() - start_eval_2
    vram_after_eval_2 = monitor.get_vram_usage_mb()
    
    # Fetch C++ timings for prompt 2 evaluation if available
    try:
        perf_data = llama_cpp.llama_perf_context(llm.ctx)
        t_p_eval_ms_2 = perf_data.t_p_eval_ms
        if t_p_eval_ms_2 > 0:
            eval_time_2 = t_p_eval_ms_2 / 1000.0
    except Exception:
        pass
        
    monitor.close()
    
    results = {
        "status": "success",
        "ambient_vram_mb": ambient_vram,
        "post_load_vram_mb": post_load_vram,
        "load_time_sec": load_time,
        "cold_eval_time_sec": eval_time_1,
        "cold_eval_vram_mb": vram_after_eval_1,
        "warm_eval_time_sec": eval_time_2,
        "warm_eval_vram_mb": vram_after_eval_2,
    }
    print(json.dumps(results))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--ngl", type=int)
    parser.add_argument("--cache", type=int) # 1 for True, 0 for False
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    
    if args.child:
        run_child_benchmark(args.model, args.ctx, args.ngl, bool(args.cache))
        sys.exit(0)
        
    downloads_dir = r"C:\Users\adars\Downloads"
    models_config = {
        "Q4_K_M": {
            "file": "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
            "ngl_configs": [33]
        },
        "Q5_K_M": {
            "file": "qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
            "ngl_configs": [33]
        },
        "Q8_0": {
            "file": "qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf",
            "ngl_configs": [33, 15]
        }
    }
    
    print("=" * 75)
    print("           PAGEDATTENTION / PROMPT CACHING COMPARATIVE SWEEPER")
    print("=" * 75)
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "json"), exist_ok=True)
    
    csv_path = os.path.join(results_dir, "paged_attention_sweeps.csv")
    csv_exists = os.path.exists(csv_path)
    
    headers = [
        "timestamp", "quantization", "ngl", "prompt_cache_enabled",
        "ambient_vram_mb", "post_load_vram_mb", "cold_prefill_time_sec",
        "cold_prefill_vram_mb", "warm_prefill_time_sec", "warm_prefill_vram_mb",
        "prefill_latency_speedup_x", "vram_allocation_delta_mb"
    ]
    
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(headers)
        csv_file.flush()
        
    all_runs_data = []
    
    # We compare both prompt_caching disabled (False) and enabled (True)
    cache_configs = [False, True]
    
    for quant, info in models_config.items():
        model_path = os.path.join(downloads_dir, info["file"])
        if not os.path.exists(model_path):
            print(f"[Warning] Model file not found for {quant}: {model_path}. Skipping.")
            continue
            
        for ngl in info["ngl_configs"]:
            print(f"\n--- Quantization: {quant} (ngl={ngl}) ---")
            for cache in cache_configs:
                print(f"  Prompt Caching Enabled = {cache}...")
                
                # Check background VRAM before run
                temp_monitor = SystemMonitor()
                ambient_check = temp_monitor.get_vram_usage_mb()
                temp_monitor.close()
                if ambient_check > 800.0:
                    print(f"    [Warning] Background VRAM high ({ambient_check:.2f} MB). Delaying for GPU release...")
                    time.sleep(2.0)
                    
                child_cmd = [
                    sys.executable, __file__,
                    "--model", model_path,
                    "--ngl", str(ngl),
                    "--cache", "1" if cache else "0",
                    "--child"
                ]
                
                try:
                    res = subprocess.run(child_cmd, capture_output=True, text=True, check=True)
                    run_metrics = json.loads(res.stdout.strip())
                    
                    cold_time = run_metrics["cold_eval_time_sec"]
                    warm_time = run_metrics["warm_eval_time_sec"]
                    
                    # Calculate speedup factor: cold_prefill_time / warm_prefill_time
                    speedup = round(cold_time / warm_time, 2) if warm_time > 0 else 0.0
                    vram_delta = round(run_metrics["warm_eval_vram_mb"] - run_metrics["post_load_vram_mb"], 2)
                    timestamp_str = datetime.now().isoformat()
                    
                    csv_row = [
                        timestamp_str, quant, ngl, str(cache),
                        round(run_metrics["ambient_vram_mb"], 2),
                        round(run_metrics["post_load_vram_mb"], 2),
                        round(cold_time, 4),
                        round(run_metrics["cold_eval_vram_mb"], 2),
                        round(warm_time, 4),
                        round(run_metrics["warm_eval_vram_mb"], 2),
                        speedup,
                        vram_delta
                    ]
                    csv_writer.writerow(csv_row)
                    csv_file.flush()
                    
                    all_runs_data.append({
                        "quant": quant,
                        "ngl": ngl,
                        "cache_enabled": cache,
                        "ambient_vram_mb": round(run_metrics["ambient_vram_mb"], 2),
                        "post_load_vram_mb": round(run_metrics["post_load_vram_mb"], 2),
                        "cold_time_sec": round(cold_time, 4),
                        "cold_vram_mb": round(run_metrics["cold_eval_vram_mb"], 2),
                        "warm_time_sec": round(warm_time, 4),
                        "warm_vram_mb": round(run_metrics["warm_eval_vram_mb"], 2),
                        "speedup_x": speedup,
                        "vram_delta_mb": vram_delta
                    })
                    
                    print(f"    Done: Cold Prefill={cold_time:.4f}s | Warm Prefill={warm_time:.4f}s | Speedup={speedup}x | VRAM Delta={vram_delta:.2f} MB")
                except Exception as e:
                    print(f"    [Error] Sweep failed: {e}")
                    
                time.sleep(1.0)
                
    csv_file.close()
    
    # Save a JSON run history file
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, "json", f"paged_attention_profile_{run_timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_runs_data, f, indent=4)
        
    print("\n" + "=" * 75)
    print("                              SWEEP SUMMARY")
    print("=" * 75)
    print(f"Results successfully saved to:")
    print(f"  CSV Database: {csv_path}")
    print(f"  JSON Artifact: {json_path}")
    print("-" * 75)
    for run in all_runs_data:
        print(f"Quant: {run['quant']} | ngl: {run['ngl']} | Cache: {str(run['cache_enabled']):5s} | Cold Prefill: {run['cold_time_sec']:.4f}s | Warm Prefill: {run['warm_time_sec']:.4f}s | Speedup: {run['speedup_x']:5.2f}x | VRAM Delta: {run['vram_delta_mb']:7.2f} MB")
    print("=" * 75)

if __name__ == "__main__":
    main()
