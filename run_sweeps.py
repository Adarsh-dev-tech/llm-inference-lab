import subprocess
import os
import sys

# Define combinations
quants = ["Q4_K_M", "Q5_K_M", "Q8_0"]
prompts = ["short", "medium", "long"]

model_paths = {
    "Q4_K_M": r"C:\Users\adars\Downloads\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    "Q5_K_M": r"C:\Users\adars\Downloads\qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
    "Q8_0": r"C:\Users\adars\Downloads\qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf"
}

python_executable = sys.executable if sys.executable else ".venv\\Scripts\\python.exe"

for quant in quants:
    model_path = model_paths[quant]
    if not os.path.exists(model_path):
        print(f"Model path for {quant} does not exist: {model_path}")
        continue
        
    for prompt_name in prompts:
        prompt_path = f"prompts/{prompt_name}.txt"
        if not os.path.exists(prompt_path):
            print(f"Prompt file {prompt_path} does not exist!")
            continue
            
        print("=" * 80)
        print(f"RUNNING SWEEP: Quant={quant} | Prompt={prompt_name}")
        print("=" * 80)
        
        # Build command
        cmd = [
            python_executable,
            "benchmark.py",
            "--model", model_path,
            "--quant", quant,
            "--prompt", prompt_path,
            "--ngl", "33",
            "--iterations", "3",
            "--max_tokens", "2048"
        ]
        
        try:
            # Run the command and capture output
            result = subprocess.run(cmd, check=True)
            print(f"Finished sweep combination successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running sweep for Quant={quant}, Prompt={prompt_name}: {e}")
            # If Q8_0 fails, maybe try with less layers offloaded? Let's check
            if quant == "Q8_0":
                print("Attempting Q8_0 with partial offloading (ngl=20)...")
                cmd_partial = cmd.copy()
                # Find --ngl and change its value to 20
                try:
                    ngl_idx = cmd_partial.index("--ngl")
                    cmd_partial[ngl_idx + 1] = "20"
                except ValueError:
                    cmd_partial.extend(["--ngl", "20"])
                try:
                    subprocess.run(cmd_partial, check=True)
                    print(f"Finished partial offload run for Q8_0.")
                except Exception as ex:
                    print(f"Failed partial offload run for Q8_0 too: {ex}")

print("All sweep combinations executed.")
