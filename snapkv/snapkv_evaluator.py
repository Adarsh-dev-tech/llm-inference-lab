import os
import sys
import time
import argparse
import json
import csv
import copy
import math
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

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import hook and quantization utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hook import patch_snapkv

# Fallback text dataset for perplexity evaluation to prevent internet dependencies
WIKITEXT_FALLBACK = """
The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. It formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud. The vast majority of the system's mass is in the Sun, with the majority of the remaining mass contained in the planet Jupiter. The four inner system planets—Mercury, Venus, Earth, and Mars—are terrestrial planets, being composed primarily of rock and metal. The four outer system planets are giant planets, being substantially more massive than the terrestrials. The two largest, Jupiter and Saturn, are gas giants, being composed mainly of hydrogen and helium; the two outermost planets, Uranus and Neptune, are ice giants, being composed mostly of substances with relatively high melting points compared with hydrogen and helium, called volatiles, such as water, ammonia and methane. All eight planets have nearly circular orbits that lie within a nearly flat disc called the ecliptic.

Large Language Models (LLMs) have revolutionized the field of natural language processing by enabling machines to understand, generate, and manipulate human language at an unprecedented scale. These models are typically based on the transformer architecture, which utilizes self-attention mechanisms to capture long-range dependencies in text. The training of LLMs involves two primary phases: pre-training on massive corpora of unlabelled text to learn general language structures, and fine-tuning on specific tasks or instruction-following datasets to align their behavior with human preferences. The computational demands of inference are a significant bottleneck for real-world deployment, especially on resource-constrained hardware such as edge devices or consumer GPUs. The key-value (KV) cache is a critical optimization that stores historical key and value states to avoid redundant calculations during autoregressive decoding. However, the size of the KV cache scales linearly with sequence length, creating a memory wall that limits the context window size on low-VRAM GPUs. Compression algorithms like SnapKV mitigate this by dynamically selecting and retaining only the most critical key-value states based on attention patterns.

In machine learning, quantization refers to the process of mapping continuous infinite values to a smaller set of discrete finite values. In the context of deep neural networks, quantization reduces the precision of weights and activations from standard 32-bit floating point (FP32) to lower-bit widths such as 8-bit integers (INT8) or 4-bit integers (INT4). This precision reduction shrinks the model's memory footprint and accelerates execution speed by leveraging specialized hardware acceleration units like Tensor Cores. However, quantization introduces quantization noise, which can degrade the model's accuracy or perplexity. Post-training quantization (PTQ) quantizes a pre-trained model directly without retraining, using a small calibration dataset to minimize precision loss. Quantization-aware training (QAT), on the other hand, models the effects of quantization during the training loop using fake-quantization modules, enabling the network parameters to adapt to precision constraints. On consumer graphics cards, custom low-bit formats like GGUF play a pivotal role in democratizing access to large models by allowing mixed CPU/GPU layer offloading.
"""

def get_vram_usage_mb() -> float:
    """Returns the current VRAM usage of the primary CUDA device in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0

def evaluate_perplexity(model, tokenizer, text: str, k: int, obs_window: int, recent_window: int, prompt_len: int = 512, gen_len: int = 64) -> float:
    """
    Computes perplexity of the model on a test prompt under SnapKV compression.
    1. Runs a prefill pass on prompt_len tokens (initializing the compressed KV cache).
    2. Runs autoregressive decoding for gen_len steps, accumulating the cross-entropy loss.
    """
    # Tokenize input text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    
    # Ensure we have enough tokens
    total_len = prompt_len + gen_len
    if input_ids.shape[1] < total_len:
        # Repeat input_ids if the text is too short
        repeats = (total_len // input_ids.shape[1]) + 1
        input_ids = input_ids.repeat(1, repeats)[:, :total_len]
        
    prompt_ids = input_ids[:, :prompt_len]
    target_ids = input_ids[:, prompt_len:total_len]
    
    # 1. Prefill stage: feed prompt to establish the compressed KV Cache
    # We patch the model with the active SnapKV configurations (or restore baseline if k == -1)
    if k == -1:
        patch_snapkv(model, disable_snapkv=True)
    else:
        patch_snapkv(model, k=k, obs_window=obs_window, recent_window=recent_window)
    
    # Prefill position IDs: [0, 1, ..., prompt_len-1]
    prefill_pos = torch.arange(prompt_len, device=model.device).unsqueeze(0)
    
    # Use standard Hugging Face past_key_values cache
    with torch.no_grad():
        outputs = model(prompt_ids, position_ids=prefill_pos, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]  # Prediction for target_ids[:, 0]
        
    # 2. Decode stage: compute cross-entropy loss step-by-step
    loss_fct = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    for i in range(gen_len):
        # Target token for this step
        target = target_ids[:, i]
        loss = loss_fct(logits, target)
        total_loss += loss.item()
        
        # Prepare logits for the next step (if not the last step)
        if i < gen_len - 1:
            current_ids = target_ids[:, i : i+1]
            # Decode position ID: [prompt_len + i]
            decode_pos = torch.tensor([[prompt_len + i]], device=model.device)
            with torch.no_grad():
                outputs = model(current_ids, position_ids=decode_pos, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :]  # Prediction for target_ids[:, i+1]
                past_key_values = outputs.past_key_values
                
    avg_loss = total_loss / gen_len
    perplexity = math.exp(avg_loss) if avg_loss < 20 else 999999.0
    return perplexity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model ID or path")
    parser.add_argument("--wikitext", type=str, default=None, help="Optional path to a Wikitext file")
    parser.add_argument("--no-snapkv", action="store_true", help="Run only the baseline perplexity evaluation without SnapKV compression")
    args = parser.parse_args()
    
    # Resolve GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print(f"Loading Tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Adjust padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load text corpus
    text = WIKITEXT_FALLBACK
    if args.wikitext and os.path.exists(args.wikitext):
        with open(args.wikitext, "r", encoding="utf-8") as f:
            text = f.read()
            
    # Setup results output
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "snapkv_benchmark.csv")
    csv_exists = os.path.exists(csv_path)
    
    headers = [
        "timestamp", "model", "precision_bits", "k_val", "observation_window", 
        "recent_window", "perplexity", "loaded_vram_mb", "prefill_peak_vram_mb"
    ]
    
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(headers)
        csv_file.flush()
        
    all_runs_data = []
    
    # Parameters for the sweep
    precisions = [4, 8]  # bits: 4-bit NF4, 8-bit Int8
    if args.no_snapkv:
        k_configs = [-1]
    else:
        k_configs = [-1, 16, 32, 64, 128, 256]
    obs_window = 32
    recent_window = 32
    
    print("\n" + "=" * 75)
    print("           SNAPKV MEMORY & ACCURACY PERPLEXITY SWEEPER")
    print("=" * 75)
    
    from transformers import BitsAndBytesConfig
    
    for bits in precisions:
        print(f"\n--- Loading Model in Actual {bits}-bit Quantization ---")
        
        # Load model in quantized format
        if device == "cuda":
            if bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                device_map = {"": 0}
            else:  # 8-bit
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                device_map = "auto"
                
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=quantization_config,
                device_map=device_map,
                attn_implementation="eager"
            )
        else:
            # CPU fallback: bitsandbytes is GPU-only
            print("[Warning] CPU detected. Loading unquantized model (bitsandbytes is GPU-only).")
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map="cpu",
                attn_implementation="eager"
            )
            
        print("Model loaded successfully.")
        
        # Record baseline memory after clean load
        torch.cuda.empty_cache()
        time.sleep(1.0)
        loaded_vram = get_vram_usage_mb()
        
        for k in k_configs:
            if k == -1:
                print("    Running baseline perplexity evaluation (SnapKV disabled)...")
            else:
                print(f"    Running SnapKV sweep with K={k}...")
            
            # Reset CUDA memory profiling
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                
            # Perform evaluation
            start_eval = time.perf_counter()
            try:
                ppl = evaluate_perplexity(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    k=k,
                    obs_window=obs_window,
                    recent_window=recent_window,
                    prompt_len=512,
                    gen_len=64
                )
                eval_time = time.perf_counter() - start_eval
                
                # Fetch peak memory usage
                if torch.cuda.is_available():
                    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
                else:
                    peak_vram = 0.0
                    
                timestamp_str = datetime.now().isoformat()
                
                # Log metrics
                csv_row = [
                    timestamp_str, os.path.basename(args.model), bits, k,
                    obs_window if k != -1 else 0, recent_window if k != -1 else 0, round(ppl, 4),
                    round(loaded_vram, 2), round(peak_vram, 2)
                ]
                csv_writer.writerow(csv_row)
                csv_file.flush()
                
                all_runs_data.append({
                    "model": os.path.basename(args.model),
                    "bits": bits,
                    "k": k,
                    "obs_window": obs_window if k != -1 else 0,
                    "recent_window": recent_window if k != -1 else 0,
                    "perplexity": round(ppl, 4),
                    "loaded_vram_mb": round(loaded_vram, 2),
                    "peak_vram_mb": round(peak_vram, 2)
                })
                
                print(f"      Done: PPL={ppl:.4f} | Loaded VRAM={loaded_vram:.2f} MB | Peak VRAM={peak_vram:.2f} MB | Time={eval_time:.2f}s")
            except Exception as e:
                print(f"      [Error] Execution failed: {e}")
                
            time.sleep(0.5)
            
        # Free memory before next load
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1.0)
            
    csv_file.close()
    
    # Save a JSON history file
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, "json", f"snapkv_profile_{run_timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_runs_data, f, indent=4)
        
    print("\n" + "=" * 75)
    print("                          SNAPKV SWEEP SUMMARY")
    print("=" * 75)
    print(f"Results successfully saved to:")
    print(f"  CSV Database: {csv_path}")
    print(f"  JSON Artifact: {json_path}")
    print("-" * 75)
    for run in all_runs_data:
        k_str = "Baseline" if run['k'] == -1 else f"K={run['k']}"
        print(f"Bits: {run['bits']} | SnapKV: {k_str:<10s} | PPL: {run['perplexity']:9.4f} | Loaded VRAM: {run['loaded_vram_mb']:7.2f} MB | Peak VRAM: {run['peak_vram_mb']:7.2f} MB")
    print("=" * 75)

if __name__ == "__main__":
    import math
    main()
