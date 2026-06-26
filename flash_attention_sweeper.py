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

# Import SystemMonitor from the workspace
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.system_monitor import SystemMonitor

def monitor_peak_vram(monitor, stop_event, peak_container):
    peak = 0.0
    while not stop_event.is_set():
        vram = monitor.get_vram_usage_mb()
        if vram > peak:
            peak = vram
        time.sleep(0.01)
    peak_container[0] = peak

def run_child_benchmark(model_path, ctx_len, use_flash, ngl):
    monitor = SystemMonitor()
    ambient_vram = monitor.get_vram_usage_mb()
    
    from llama_cpp import Llama
    
    start_load = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=ngl,
        n_ctx=ctx_len,
        flash_attn=use_flash,
        verbose=False
    )
    load_time = time.perf_counter() - start_load
    
    post_load_vram = monitor.get_vram_usage_mb()
    
    # Warmup
    warmup_tokens = [llm.token_bos()] + [100] * 3
    llm.eval(warmup_tokens)
    vram_after_warmup = monitor.get_vram_usage_mb()
    
    # Prefill tokens (ctx_len - 10 to leave space for decode)
    prefill_size = ctx_len - 10
    prefill_tokens = [llm.token_bos()] + [100] * (prefill_size - 1)
    
    llm.reset()
    
    stop_event = threading.Event()
    peak_container = [vram_after_warmup]
    monitor_thread = threading.Thread(
        target=monitor_peak_vram, 
        args=(monitor, stop_event, peak_container), 
        daemon=True
    )
    monitor_thread.start()
    
    start_prefill = time.perf_counter()
    llm.eval(prefill_tokens)
    prefill_time = time.perf_counter() - start_prefill
    
    time.sleep(0.2)
    stop_event.set()
    monitor_thread.join(timeout=1.0)
    
    prefill_peak_vram = peak_container[0]
    
    # Decode 5 tokens
    start_decode = time.perf_counter()
    current_token = 100
    for _ in range(5):
        llm.eval([current_token])
        current_token = 101
    decode_time = (time.perf_counter() - start_decode) / 5
    
    post_decode_vram = monitor.get_vram_usage_mb()
    monitor.close()
    
    results = {
        "status": "success",
        "ambient_vram_mb": ambient_vram,
        "post_load_vram_mb": post_load_vram,
        "prefill_peak_vram_mb": prefill_peak_vram,
        "prefill_spike_mb": max(0.0, prefill_peak_vram - post_load_vram),
        "post_decode_vram_mb": post_decode_vram,
        "load_time_sec": load_time,
        "prefill_time_sec": prefill_time,
        "decode_step_time_sec": decode_time,
    }
    print(json.dumps(results))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ctx", type=int)
    parser.add_argument("--flash", type=int) # 1 for True, 0 for False
    parser.add_argument("--ngl", type=int)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    
    if args.child:
        run_child_benchmark(args.model, args.ctx, bool(args.flash), args.ngl)
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
    
    context_lengths = [512, 1024, 2048, 4096]
    flash_configs = [False, True]
    
    print("=" * 70)
    print("      FLASH ATTENTION COMPARATIVE SWEEPER (ALL QUANTS)")
    print("=" * 70)
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "json"), exist_ok=True)
    
    csv_path = os.path.join(results_dir, "flash_attn_sweeps.csv")
    csv_exists = os.path.exists(csv_path)
    
    headers = [
        "timestamp", "quantization", "ngl", "flash_attn", "context_length",
        "ambient_vram_mb", "post_load_vram_mb", "prefill_peak_vram_mb",
        "prefill_spike_mb", "post_decode_vram_mb", "prefill_time_sec",
        "decode_speed_tps", "is_paging_triggered"
    ]
    
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(headers)
        csv_file.flush()
        
    all_runs_data = []
    
    for quant, info in models_config.items():
        model_path = os.path.join(downloads_dir, info["file"])
        if not os.path.exists(model_path):
            print(f"[Warning] Model file not found for {quant}: {model_path}. Skipping.")
            continue
            
        for ngl in info["ngl_configs"]:
            print(f"\n--- Quantization: {quant} (ngl={ngl}) ---")
            for flash in flash_configs:
                print(f"  FlashAttention = {flash}...")
                for ctx in context_lengths:
                    print(f"    Running context length {ctx}...")
                    
                    # Verify background usage before each run
                    temp_monitor = SystemMonitor()
                    ambient_check = temp_monitor.get_vram_usage_mb()
                    temp_monitor.close()
                    if ambient_check > 800.0:
                        print(f"      [Warning] Ambient VRAM high ({ambient_check:.2f} MB). Delaying for GPU release...")
                        time.sleep(2.0)
                        
                    child_cmd = [
                        sys.executable, __file__, 
                        "--model", model_path, 
                        "--ctx", str(ctx), 
                        "--flash", "1" if flash else "0", 
                        "--ngl", str(ngl),
                        "--child"
                    ]
                    
                    try:
                        res = subprocess.run(child_cmd, capture_output=True, text=True, check=True)
                        run_metrics = json.loads(res.stdout.strip())
                        
                        is_paging = "Yes" if run_metrics["prefill_peak_vram_mb"] > 6000.0 else "No"
                        decode_speed = round(1.0 / run_metrics["decode_step_time_sec"], 2)
                        timestamp_str = datetime.now().isoformat()
                        
                        csv_row = [
                            timestamp_str, quant, ngl, str(flash), ctx,
                            round(run_metrics["ambient_vram_mb"], 2),
                            round(run_metrics["post_load_vram_mb"], 2),
                            round(run_metrics["prefill_peak_vram_mb"], 2),
                            round(run_metrics["prefill_spike_mb"], 2),
                            round(run_metrics["post_decode_vram_mb"], 2),
                            round(run_metrics["prefill_time_sec"], 4),
                            decode_speed,
                            is_paging
                        ]
                        csv_writer.writerow(csv_row)
                        csv_file.flush()
                        
                        all_runs_data.append({
                            "quant": quant,
                            "ngl": ngl,
                            "flash": flash,
                            "ctx": ctx,
                            "ambient_vram_mb": round(run_metrics["ambient_vram_mb"], 2),
                            "post_load_vram_mb": round(run_metrics["post_load_vram_mb"], 2),
                            "prefill_peak_vram_mb": round(run_metrics["prefill_peak_vram_mb"], 2),
                            "prefill_spike_mb": round(run_metrics["prefill_spike_mb"], 2),
                            "post_decode_vram_mb": round(run_metrics["post_decode_vram_mb"], 2),
                            "prefill_time_sec": round(run_metrics["prefill_time_sec"], 4),
                            "decode_speed_tps": decode_speed,
                            "is_paging": is_paging
                        })
                        
                        print(f"      Done: Loaded VRAM={run_metrics['post_load_vram_mb']:.2f} MB | Prefill Peak={run_metrics['prefill_peak_vram_mb']:.2f} MB | Prefill Time={run_metrics['prefill_time_sec']:.4f}s | Decode={decode_speed:.2f} t/s | Paging={is_paging}")
                    except Exception as e:
                        print(f"      [Error] Failed run: {e}")
                        
                    time.sleep(1.0)
            
    csv_file.close()
    
    # Save a JSON history run file
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, "json", f"flash_attn_profile_{run_timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_runs_data, f, indent=4)
        
    print("\n" + "=" * 70)
    print("                              SWEEP SUMMARY")
    print("=" * 70)
    print(f"Results successfully saved to:")
    print(f"  CSV Database: {csv_path}")
    print(f"  JSON Artifact: {json_path}")
    print("-" * 70)
    for run in all_runs_data:
        print(f"Quant: {run['quant']} | ngl: {run['ngl']} | Flash: {str(run['flash']):5s} | Ctx: {run['ctx']:4d} | Loaded VRAM: {run['post_load_vram_mb']:7.2f} MB | Prefill Peak: {run['prefill_peak_vram_mb']:7.2f} MB | Prefill Spike: {run['prefill_spike_mb']:7.2f} MB | Prefill Time: {run['prefill_time_sec']:7.4f}s | Decode: {run['decode_speed_tps']:6.2f} t/s | Paging: {run['is_paging']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
