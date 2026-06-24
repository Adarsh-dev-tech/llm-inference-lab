import os
import sys
import time
import argparse
import subprocess
import json
import csv
import threading
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
try:
    from utils.system_monitor import SystemMonitor
except ImportError:
    # Fallback to direct import if Cwd is different
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils.system_monitor import SystemMonitor


# Qwen2.5-7B Architectural Constants for theoretical calculations
QWEN_LAYERS = 28
QWEN_HEADS_Q = 28
QWEN_HEADS_KV = 4  # Grouped-Query Attention (GQA)
QWEN_HEAD_DIM = 128
BYTES_PER_FP16_ELEMENT = 2

def calculate_theoretical_kv_bytes(ctx_len, layers=QWEN_LAYERS, heads_kv=QWEN_HEADS_KV, head_dim=QWEN_HEAD_DIM, bytes_per_element=BYTES_PER_FP16_ELEMENT):
    """
    Theoretical KV Cache formula:
    Size = 2 * layers * heads_kv * head_dim * bytes_per_element * context_len
    """
    return 2 * layers * heads_kv * head_dim * bytes_per_element * ctx_len

def monitor_peak_vram(monitor, stop_event, peak_container):
    """Worker function to sample VRAM at high frequency to catch the prefill spike."""
    peak = 0.0
    while not stop_event.is_set():
        vram = monitor.get_vram_usage_mb()
        if vram > peak:
            peak = vram
        time.sleep(0.02)
    peak_container[0] = peak

def run_child_benchmark(model_path, quant, ctx_len, ngl):
    """
    Executes a single measurement run. This function is called when running as a child process.
    """
    monitor = SystemMonitor()
    
    # Measure ambient baseline VRAM before doing anything
    ambient_vram = monitor.get_vram_usage_mb()
    
    # 1. Load the model
    from llama_cpp import Llama
    
    start_load = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=ngl,
        n_ctx=ctx_len,
        verbose=False
    )
    load_time = time.perf_counter() - start_load
    
    post_load_vram = monitor.get_vram_usage_mb()
    loaded_delta = max(0.0, post_load_vram - ambient_vram)
    
    # 2. Run warmup (small token generation) to initialize CUDA context fully
    # We evaluate 4 tokens to avoid polluting prefill measurements
    warmup_tokens = [llm.token_bos()] + [100] * 3
    llm.eval(warmup_tokens)
    
    vram_after_warmup = monitor.get_vram_usage_mb()
    
    # 3. Create prefill sequence slightly smaller than the context limit
    # to leave room for the subsequent decoding steps in the pre-allocated cache.
    prefill_size = ctx_len - 10
    prefill_tokens = [llm.token_bos()] + [100] * (prefill_size - 1)
    
    # Reset KV cache context state in llama.cpp for clean prefill evaluation
    llm.reset()
    
    # Set up high-frequency VRAM monitor to capture the prefill activation spike
    stop_event = threading.Event()
    peak_container = [vram_after_warmup]
    monitor_thread = threading.Thread(
        target=monitor_peak_vram, 
        args=(monitor, stop_event, peak_container), 
        daemon=True
    )
    monitor_thread.start()
    
    # Run the prefill stage
    start_prefill = time.perf_counter()
    llm.eval(prefill_tokens)
    prefill_time = time.perf_counter() - start_prefill
    
    # Let VRAM stabilize slightly, then stop background peak monitor
    time.sleep(0.2)
    stop_event.set()
    monitor_thread.join(timeout=1.0)
    
    prefill_peak_vram = peak_container[0]
    
    # 4. Measure steady-state decoding VRAM
    # Generate 5 tokens autoregressively
    start_decode = time.perf_counter()
    decoded_tokens = []
    current_token = 100
    for _ in range(5):
        # Eval a single token
        llm.eval([current_token])
        # Extremely simplified token selection (just mock next token)
        current_token = 101
    decode_time = (time.perf_counter() - start_decode) / 5
    
    post_decode_vram = monitor.get_vram_usage_mb()
    
    # Cleanup NVML
    monitor.close()
    
    # Package output metrics
    results = {
        "status": "success",
        "ambient_vram_mb": ambient_vram,
        "post_load_vram_mb": post_load_vram,
        "loaded_delta_mb": loaded_delta,
        "prefill_peak_vram_mb": prefill_peak_vram,
        "prefill_spike_mb": max(0.0, prefill_peak_vram - post_load_vram),
        "post_decode_vram_mb": post_decode_vram,
        "load_time_sec": load_time,
        "prefill_time_sec": prefill_time,
        "decode_step_time_sec": decode_time,
    }
    
    # Output to stdout as JSON so parent can capture it
    print(json.dumps(results))


def main():
    parser = argparse.ArgumentParser(description="Isolated KV Cache Memory Profiler")
    parser.add_argument("--model", type=str, help="Path to GGUF model")
    parser.add_argument("--quant", type=str, help="Model quantization format")
    parser.add_argument("--ctx", type=int, help="Context length")
    parser.add_argument("--ngl", type=int, help="Number of GPU layers offloaded")
    parser.add_argument("--child", action="store_true", help="Run in subprocess child mode")
    
    args = parser.parse_args()
    
    if args.child:
        if not args.model or not args.quant or not args.ctx or args.ngl is None:
            print(json.dumps({"status": "error", "message": "Missing arguments in child mode"}))
            sys.exit(1)
        try:
            run_child_benchmark(args.model, args.quant, args.ctx, args.ngl)
        except Exception as e:
            import traceback
            print(json.dumps({
                "status": "error", 
                "message": str(e), 
                "trace": traceback.format_exc()
            }))
            sys.exit(1)
        sys.exit(0)
        
    # Main parent orchestrator mode
    print("=" * 70)
    print("            KV CACHE MEMORY PROFILER SYSTEM")
    print("=" * 70)
    
    # Model files lookup
    downloads_dir = r"C:\Users\adars\Downloads"
    models_config = {
        "Q4_K_M": {
            "file": "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
            "base_file_name": "qwen2.5-7b-instruct-q4_k_m-00001-of-00002"
        },
        "Q5_K_M": {
            "file": "qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
            "base_file_name": "qwen2.5-7b-instruct-q5_k_m-00001-of-00002"
        },
        "Q8_0": {
            "file": "qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf",
            "base_file_name": "qwen2.5-7b-instruct-q8_0-00001-of-00003"
        }
    }
    
    # Generate context sweeps
    context_lengths = [128, 256, 512, 1024, 2048, 4096]
    
    # Prepare results storage
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "json"), exist_ok=True)
    
    csv_path = os.path.join(results_dir, "kv_cache_growth.csv")
    csv_exists = os.path.exists(csv_path)
    
    # CSV headers matching detailed hardware and allocation dimensions
    headers = [
        "timestamp", "quantization", "context_length", "ngl", 
        "theoretical_kv_mb", "ambient_vram_mb", "post_load_vram_mb", 
        "loaded_delta_mb", "prefill_peak_vram_mb", "prefill_spike_mb", 
        "post_decode_vram_mb", "experimental_kv_estimate_mb", 
        "is_paging_triggered", "load_time_sec", "prefill_time_sec", 
        "decode_step_time_sec"
    ]
    
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(headers)
        csv_file.flush()
        
    all_runs_data = []
    
    # Execute loops
    for quant, info in models_config.items():
        model_path = os.path.join(downloads_dir, info["file"])
        if not os.path.exists(model_path):
            print(f"[Warning] Model file not found for {quant}: {model_path}. Skipping.")
            continue
            
        print(f"\n--- Profiling Model Quantization: {quant} ---")
        
        # Decide offloading configurations (Strategy A: Full ngl=33; Strategy B: Optimized partial offloading if needed)
        offload_configs = [33]
        
        # SPECIAL CASE: For Q8_0 model, it is too large (~7.7 GB weights) to fit in 6GB VRAM.
        # Running it with ngl=33 triggers heavy PCIe memory paging, resulting in slow 3.8-5 tps.
        # We also run a partial offload config (ngl=15 layers) as a "special case" comparison.
        # This keeps offloaded weights under ~4.1 GB and fits completely inside the physical VRAM.
        if quant == "Q8_0":
            offload_configs = [33, 15]
            print("[Special Case] For Q8_0, we will benchmark both ngl=33 (Full Paging) and ngl=15 (Optimized VRAM Fallback).")
            
        for ngl in offload_configs:
            # We first load a minimal baseline (n_ctx=8) for this model/ngl combination.
            # This isolates the overhead of weights + graph buffers, so we can calculate
            # experimental KV cache sizes accurately by subtracting it from larger context loads.
            print(f"  Establishing weights-only baseline (n_ctx=8) with ngl={ngl}...")
            baseline_cmd = [
                sys.executable, __file__, 
                "--model", model_path, 
                "--quant", quant, 
                "--ctx", "8", 
                "--ngl", str(ngl), 
                "--child"
            ]
            
            try:
                res = subprocess.run(baseline_cmd, capture_output=True, text=True, check=True)
                child_data = json.loads(res.stdout.strip())
                baseline_loaded_vram = child_data["post_load_vram_mb"]
                print(f"    Weights-only baseline VRAM: {baseline_loaded_vram:.2f} MB")
            except Exception as e:
                print(f"    [Error] Failed to establish baseline for {quant} ngl={ngl}: {e}. Skipping config.")
                continue

            for ctx in context_lengths:
                print(f"  Running context length {ctx} (ngl={ngl})...")
                
                # Check for other processes before starting (double check baseline remains clean)
                temp_monitor = SystemMonitor()
                ambient_check = temp_monitor.get_vram_usage_mb()
                temp_monitor.close()
                if ambient_check > 800.0:
                    print(f"    [Warning] Background GPU memory is currently high ({ambient_check:.2f} MB). VRAM metrics might drift.")
                
                child_cmd = [
                    sys.executable, __file__, 
                    "--model", model_path, 
                    "--quant", quant, 
                    "--ctx", str(ctx), 
                    "--ngl", str(ngl), 
                    "--child"
                ]
                
                start_run = time.perf_counter()
                try:
                    res = subprocess.run(child_cmd, capture_output=True, text=True, check=True)
                    run_duration = time.perf_counter() - start_run
                    
                    # Parse child outputs
                    child_output = res.stdout.strip()
                    run_metrics = json.loads(child_output)
                    
                    if run_metrics.get("status") == "error":
                        print(f"    [Subprocess Error] {run_metrics.get('message')}")
                        continue
                        
                    # Calculate derived metrics
                    theoretical_kv_bytes = calculate_theoretical_kv_bytes(ctx)
                    theoretical_kv_mb = round(theoretical_kv_bytes / (1024 * 1024), 2)
                    
                    # Experimental KV cache is computed as the increase in loaded VRAM
                    # compared to the minimal (n_ctx=8) baseline load for this offload level.
                    experimental_kv = max(0.0, run_metrics["post_load_vram_mb"] - baseline_loaded_vram)
                    
                    # Paging indicator: if total VRAM footprint exceeds 6,144 MB physical limit.
                    is_paging = "Yes" if run_metrics["prefill_peak_vram_mb"] > 6000.0 else "No"
                    
                    timestamp_str = datetime.now().isoformat()
                    
                    # Log to CSV
                    csv_row = [
                        timestamp_str, quant, ctx, ngl, 
                        theoretical_kv_mb, 
                        round(run_metrics["ambient_vram_mb"], 2),
                        round(run_metrics["post_load_vram_mb"], 2),
                        round(run_metrics["loaded_delta_mb"], 2),
                        round(run_metrics["prefill_peak_vram_mb"], 2),
                        round(run_metrics["prefill_spike_mb"], 2),
                        round(run_metrics["post_decode_vram_mb"], 2),
                        round(experimental_kv, 2),
                        is_paging,
                        round(run_metrics["load_time_sec"], 4),
                        round(run_metrics["prefill_time_sec"], 4),
                        round(run_metrics["decode_step_time_sec"], 4)
                    ]
                    
                    csv_writer.writerow(csv_row)
                    csv_file.flush()
                    
                    # Store in-memory for parent summary
                    all_runs_data.append({
                        "quant": quant, "ctx": ctx, "ngl": ngl,
                        "theoretical_kv_mb": theoretical_kv_mb,
                        "experimental_kv_mb": round(experimental_kv, 2),
                        "prefill_peak": round(run_metrics["prefill_peak_vram_mb"], 2),
                        "prefill_spike": round(run_metrics["prefill_spike_mb"], 2),
                        "prefill_time": round(run_metrics["prefill_time_sec"], 4),
                        "decode_time": round(run_metrics["decode_step_time_sec"], 4),
                        "is_paging": is_paging
                    })
                    
                    print(f"    Done: Loaded VRAM={run_metrics['post_load_vram_mb']:.2f} MB | Prefill Peak={run_metrics['prefill_peak_vram_mb']:.2f} MB | Prefill Time={run_metrics['prefill_time_sec']:.4f}s | Decode={1.0/run_metrics['decode_step_time_sec']:.2f} t/s | Paging={is_paging}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"    [Execution Failure] Child process returned code {e.returncode}")
                    print(f"    Stderr: {e.stderr}")
                except json.JSONDecodeError:
                    print(f"    [Execution Failure] Could not parse child output as JSON.")
                    print(f"    Raw Output: {res.stdout}")
                except Exception as e:
                    print(f"    [Unexpected Error] {e}")
                    
                # Short cooldown sleep between runs to allow OS/GPU to clean up memory
                time.sleep(1.0)
                
    csv_file.close()
    
    # Save a JSON run history artifact
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, "json", f"kv_cache_profile_{run_timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_runs_data, f, indent=4)
        
    print("\n" + "=" * 70)
    print("                      PROFILING RUN SUMMARY")
    print("=" * 70)
    print(f"Results successfully saved to:")
    print(f"  CSV Database: {csv_path}")
    print(f"  JSON Artifact: {json_path}")
    print("-" * 70)
    for run in all_runs_data:
        print(f"Quant: {run['quant']} | Ctx: {run['ctx']:4d} | ngl: {run['ngl']} | Theor KV: {run['theoretical_kv_mb']:6.2f} MB | Exp KV: {run['experimental_kv_mb']:6.2f} MB | Prefill Spike: {run['prefill_spike']:6.2f} MB | Decode Speed: {1.0/run['decode_time']:6.2f} t/s | Paging: {run['is_paging']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
