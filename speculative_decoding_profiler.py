import os
import sys
import time
import subprocess
import json
import csv
import re
import threading
from datetime import datetime

# Ensure CUDA DLLs are found on Windows
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

def monitor_peak_vram(monitor, stop_event, peak_container):
    """Worker function to sample VRAM at high frequency to catch execution spike."""
    peak = 0.0
    while not stop_event.is_set():
        vram = monitor.get_vram_usage_mb()
        if vram > peak:
            peak = vram
        time.sleep(0.05)
    peak_container[0] = peak

def run_speculative_benchmark(cli_path, target_path, draft_path, ngl, ngld, use_spec, ctx_len=2048, prompt="Explain backpropagation in one sentence.", gen_len=64):
    """
    Runs a single execution of llama-cli to evaluate speculative decoding.
    Returns: (prompt_tps, gen_tps, acceptance_rate, peak_vram, load_time)
    """
    monitor = SystemMonitor()
    ambient_vram = monitor.get_vram_usage_mb()
    
    cmd = [
        cli_path,
        "-m", target_path,
        "-ngl", str(ngl),
        "-p", prompt,
        "-n", str(gen_len),
        "-st",
        "-c", str(ctx_len),
        "--verbose"
    ]
    
    if use_spec:
        cmd.extend([
            "-md", draft_path,
            "-ngld", str(ngld),
            "--spec-type", "draft-simple"
        ])
        
    start_time = time.perf_counter()
    
    # Start VRAM profiling thread
    stop_event = threading.Event()
    peak_container = [ambient_vram]
    monitor_thread = threading.Thread(target=monitor_peak_vram, args=(monitor, stop_event, peak_container))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Run the llama-cli subprocess
    try:
        # Run with shell=True/False depending on needs. False is cleaner.
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        execution_time = time.perf_counter() - start_time
        stdout = result.stdout
        stderr = result.stderr
        output = stdout + "\n" + stderr
    finally:
        # Stop VRAM monitoring
        stop_event.set()
        monitor_thread.join()
        monitor.close()
        
    # Parse prompt and generation throughput
    # Look for: [ Prompt: 345.3 t/s | Generation: 37.2 t/s ]
    prompt_tps = 0.0
    gen_tps = 0.0
    tps_match = re.search(r"Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s", output)
    if tps_match:
        prompt_tps = float(tps_match.group(1))
        gen_tps = float(tps_match.group(2))
    else:
        # Fallback to verbose timing lines:
        # prompt eval time =     101.36 ms /    35 tokens (   276.13 tokens per second)
        prompt_eval_match = re.search(r"prompt eval time\s*=.*?([\d\.]+)\s*tokens per second", output)
        if prompt_eval_match:
            prompt_tps = float(prompt_eval_match.group(1))
            
        eval_match = re.search(r"eval time\s*=.*?([\d\.]+)\s*tokens per second", output)
        if eval_match:
            gen_tps = float(eval_match.group(1))
            
    # Parse draft acceptance rate
    # Look for: draft acceptance = 0.50000
    acceptance_rate = None
    acc_match = re.search(r"draft acceptance\s*=\s*([\d\.]+)", output)
    if acc_match:
        acceptance_rate = float(acc_match.group(1))
        
    # Parse load time
    # Look for: load time = 1903.30 ms
    load_time = 0.0
    load_match = re.search(r"load time\s*=\s*([\d\.]+)\s*ms", output)
    if load_match:
        load_time = float(load_match.group(1)) / 1000.0
    else:
        load_time = execution_time # Fallback to total run time if not found
        
    peak_vram = peak_container[0]
    
    return {
        "prompt_tps": prompt_tps,
        "gen_tps": gen_tps,
        "acceptance_rate": acceptance_rate,
        "peak_vram_mb": round(peak_vram, 2),
        "load_time_sec": round(load_time, 3),
        "execution_time_sec": round(execution_time, 2)
    }

def main():
    cli_path = r"C:\Users\adars\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-cli.exe"
    downloads_dir = r"C:\Users\adars\Downloads"
    
    draft_file = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    draft_path = os.path.join(downloads_dir, draft_file)
    
    targets_config = {
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
    
    if not os.path.exists(cli_path):
        print(f"[Error] llama-cli.exe not found at: {cli_path}")
        sys.exit(1)
        
    if not os.path.exists(draft_path):
        print(f"[Error] Draft model not found at: {draft_path}")
        sys.exit(1)
        
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "json"), exist_ok=True)
    
    csv_path = os.path.join(results_dir, "speculative_decoding_benchmark.csv")
    csv_exists = os.path.exists(csv_path)
    
    headers = [
        "timestamp", "target_quantization", "target_ngl", "speculative_enabled", "draft_model",
        "prompt_tps", "generation_tps", "acceptance_rate", "peak_vram_mb", 
        "load_time_sec", "total_run_time_sec"
    ]
    
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(headers)
        csv_file.flush()
        
    all_runs = []
    
    print("=" * 80)
    print("           NATIVE SPECULATIVE DECODING COMPARATIVE SWEEPER")
    print("=" * 80)
    
    # We run disabled (False) then enabled (True)
    spec_configs = [False, True]
    
    for quant, info in targets_config.items():
        target_path = os.path.join(downloads_dir, info["file"])
        if not os.path.exists(target_path):
            print(f"[Warning] Target model file not found for {quant}: {target_path}. Skipping.")
            continue
            
        for ngl in info["ngl_configs"]:
            print(f"\n--- Target Model: {quant} (ngl={ngl}) ---")
            for use_spec in spec_configs:
                spec_str = "ENABLED" if use_spec else "DISABLED"
                print(f"  Speculative Decoding = {spec_str}...")
                
                # Check VRAM load and add small cooling sleep
                temp_monitor = SystemMonitor()
                ambient = temp_monitor.get_vram_usage_mb()
                temp_monitor.close()
                if ambient > 1000.0:
                    print(f"    [Warning] VRAM in use is high ({ambient:.2f} MB). Waiting for cooldown...")
                    time.sleep(3.0)
                    
                # Run the benchmark
                metrics = run_speculative_benchmark(
                    cli_path=cli_path,
                    target_path=target_path,
                    draft_path=draft_path,
                    ngl=ngl,
                    ngld=33 if use_spec else 0,
                    use_spec=use_spec,
                    ctx_len=2048,
                    prompt="Explain backpropagation in one sentence.",
                    gen_len=64
                )
                
                timestamp_str = datetime.now().isoformat()
                acc_rate_str = str(metrics["acceptance_rate"]) if metrics["acceptance_rate"] is not None else "N/A"
                draft_name = os.path.basename(draft_path) if use_spec else "None"
                
                # Write to CSV
                csv_row = [
                    timestamp_str, quant, ngl, str(use_spec), draft_name,
                    metrics["prompt_tps"], metrics["gen_tps"], acc_rate_str,
                    metrics["peak_vram_mb"], metrics["load_time_sec"], metrics["execution_time_sec"]
                ]
                csv_writer.writerow(csv_row)
                csv_file.flush()
                
                all_runs.append({
                    "target_quant": quant,
                    "target_ngl": ngl,
                    "speculative_enabled": use_spec,
                    "prompt_tps": metrics["prompt_tps"],
                    "generation_tps": metrics["gen_tps"],
                    "acceptance_rate": metrics["acceptance_rate"],
                    "peak_vram_mb": metrics["peak_vram_mb"],
                    "load_time_sec": metrics["load_time_sec"],
                    "total_run_time_sec": metrics["execution_time_sec"]
                })
                
                print(f"    Done: Prompt={metrics['prompt_tps']:.1f} t/s | Gen={metrics['gen_tps']:.1f} t/s | Acc={acc_rate_str} | Peak VRAM={metrics['peak_vram_mb']:.1f} MB")
                time.sleep(2.0)
                
    csv_file.close()
    
    # Save a JSON run history file
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, "json", f"speculative_decoding_profile_{run_timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=4)
        
    print("\n" + "=" * 80)
    print("                          SWEEP SUMMARY")
    print("=" * 80)
    print(f"Results successfully saved to:")
    print(f"  CSV Database: {csv_path}")
    print(f"  JSON Artifact: {json_path}")
    print("-" * 80)
    for run in all_runs:
        acc_str = f"{run['acceptance_rate'] * 100:.1f}%" if run['acceptance_rate'] is not None else "N/A"
        print(f"Target: {run['target_quant']} (ngl={run['target_ngl']:2d}) | Speculative: {str(run['speculative_enabled']):5s} | Prompt: {run['prompt_tps']:6.1f} t/s | Gen: {run['generation_tps']:5.1f} t/s | Acc: {acc_str:6s} | Peak VRAM: {run['peak_vram_mb']:7.1f} MB")
    print("=" * 80)

if __name__ == "__main__":
    main()
