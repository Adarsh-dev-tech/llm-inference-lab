import os
import csv
import json
from datetime import datetime

def log_to_csv(csv_path: str, data: dict):
    """
    Appends a benchmark run's metrics to the CSV file.
    Creates the file and writes the header if it doesn't exist.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Consistent header structure with all extended metrics.
    headers = [
        "run_id",
        "iteration_index",
        "timestamp",
        "model",
        "quantization",
        "n_gpu_layers",
        "prompt_tokens",
        "generated_tokens",
        "prompt_tps",
        "generation_tps",
        "first_token_latency_ms",
        "total_request_latency_ms",
        "ram_before_mb",
        "ram_after_mb",
        "vram_before_mb",
        "vram_after_mb",
        "context_length",
        "total_generation_time_ms",
        "cpu_utilization",
        "gpu_utilization",
        "avg_itl_ms",
        "process_ram_before_mb",
        "process_ram_after_mb",
        "gpu_mem_utilization",
        "gpu_temperature_c",
        "gpu_power_watts",
        "gpu_graphics_clock_mhz",
        "gpu_memory_clock_mhz",
        "ram_before_load_mb",
        "ram_after_load_mb",
        "vram_before_load_mb",
        "vram_after_load_mb"
    ]
    
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        
        # Prepare the row with fallback defaults for missing keys
        row = {h: data.get(h, "") for h in headers}
        writer.writerow(row)

def save_json_artifact(json_dir: str, consolidated_data: dict, run_datetime: datetime) -> str:
    """
    Saves the complete benchmark snapshot (including multiple iterations, statistics summaries,
    hardware specs, and raw execution details) as a timestamped JSON file.
    Naming format: YYYY-MM-DD_HH-MM-SS.json
    """
    os.makedirs(json_dir, exist_ok=True)
    filename = run_datetime.strftime("%Y-%m-%d_%H-%M-%S.json")
    filepath = os.path.join(json_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(consolidated_data, f, indent=2)
        
    return filepath
