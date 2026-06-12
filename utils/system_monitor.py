import threading
import time
import subprocess
import os
import platform
import winreg
import psutil

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

class SystemMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.cpu_samples = []
        self.gpu_samples = []
        self.gpu_mem_samples = []
        self.gpu_temp_samples = []
        self.gpu_power_samples = []
        self.gpu_clock_graphics_samples = []
        self.gpu_clock_mem_samples = []
        
        self.monitoring = False
        self.thread = None
        self._nvml_initialized = False
        
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception:
                pass

    def get_cpu_name(self) -> str:
        """Retrieves the CPU name dynamically from the Windows Registry."""
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            return name.strip()
        except Exception:
            return platform.processor() or "Intel Core i5 (12th Gen)"

    def get_total_ram_gb(self) -> float:
        """Returns the total host RAM in GB."""
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)

    def get_ram_usage_mb(self) -> float:
        """Returns the current host RAM usage in MB."""
        return round(psutil.virtual_memory().used / (1024 * 1024), 2)

    def get_process_ram_usage_mb(self) -> float:
        """Returns the resident set size (RSS) memory of the current process in MB."""
        try:
            return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
        except Exception:
            return 0.0

    def get_gpu_name(self) -> str:
        """Retrieves the GPU name dynamically from NVML or nvidia-smi."""
        if self._nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return pynvml.nvmlDeviceGetName(handle)
            except Exception:
                pass
        
        # Fallback to nvidia-smi
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return res.stdout.strip()
        except Exception:
            return "NVIDIA GeForce RTX 3050 Laptop GPU"

    def get_total_vram_gb(self) -> float:
        """Returns the total VRAM in GB."""
        if self._nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return round(info.total / (1024 ** 3), 1)
            except Exception:
                pass
        
        # Fallback to nvidia-smi
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return round(float(res.stdout.strip()) / 1024, 1)
        except Exception:
            return 6.0

    def get_vram_usage_mb(self) -> float:
        """Returns the current VRAM usage of GPU 0 in MB."""
        if self._nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return round(info.used / (1024 * 1024), 2)
            except Exception:
                pass
        
        # Fallback to nvidia-smi
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return float(res.stdout.strip())
        except Exception:
            return 0.0

    def get_gpu_utilization(self) -> float:
        """Returns instantaneous GPU utilization percentage."""
        if self._nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return float(util.gpu)
            except Exception:
                pass
        
        # Fallback to nvidia-smi
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return float(res.stdout.strip())
        except Exception:
            return 0.0

    def get_gpu_memory_utilization(self) -> float:
        """Returns memory controller utilization percentage (memory bandwidth utilization)."""
        if self._nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return float(util.memory)
            except Exception:
                pass
        
        # Fallback to nvidia-smi
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.memory", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return float(res.stdout.strip())
        except Exception:
            return 0.0

    def get_gpu_temperature(self) -> float:
        """Returns the current GPU temperature in Celsius."""
        if self._nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
        
        # Fallback to nvidia-smi
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return float(res.stdout.strip())
        except Exception:
            return 0.0

    def get_gpu_graphics_clock(self) -> float:
        """Returns the current GPU graphics clock in MHz."""
        if self._nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
            except Exception:
                pass
        
        # Fallback to nvidia-smi
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.gr", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return float(res.stdout.strip())
        except Exception:
            return 0.0

    def get_gpu_memory_clock(self) -> float:
        """Returns the current GPU memory clock in MHz."""
        if self._nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
            except Exception:
                pass
        
        # Fallback to nvidia-smi
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.mem", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return float(res.stdout.strip())
        except Exception:
            return 0.0

    def get_gpu_power_draw(self) -> float:
        """Returns the current GPU power usage in Watts."""
        if self._nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
            except Exception:
                pass
        
        # Fallback to nvidia-smi
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return float(res.stdout.strip())
        except Exception:
            return 0.0

    def _monitor_loop(self):
        """Monitors CPU and GPU statistics in a background thread."""
        # Prime the CPU utilization counter
        psutil.cpu_percent(interval=None)
        
        while self.monitoring:
            try:
                self.cpu_samples.append(psutil.cpu_percent(interval=None))
                self.gpu_samples.append(self.get_gpu_utilization())
                self.gpu_mem_samples.append(self.get_gpu_memory_utilization())
                self.gpu_temp_samples.append(self.get_gpu_temperature())
                self.gpu_power_samples.append(self.get_gpu_power_draw())
                self.gpu_clock_graphics_samples.append(self.get_gpu_graphics_clock())
                self.gpu_clock_mem_samples.append(self.get_gpu_memory_clock())
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        """Starts background monitoring."""
        self.cpu_samples = []
        self.gpu_samples = []
        self.gpu_mem_samples = []
        self.gpu_temp_samples = []
        self.gpu_power_samples = []
        self.gpu_clock_graphics_samples = []
        self.gpu_clock_mem_samples = []
        
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict:
        """Stops background monitoring and returns the aggregated average stats."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        avg_gpu = sum(self.gpu_samples) / len(self.gpu_samples) if self.gpu_samples else 0.0
        avg_gpu_mem = sum(self.gpu_mem_samples) / len(self.gpu_mem_samples) if self.gpu_mem_samples else 0.0
        avg_gpu_temp = sum(self.gpu_temp_samples) / len(self.gpu_temp_samples) if self.gpu_temp_samples else 0.0
        avg_gpu_power = sum(self.gpu_power_samples) / len(self.gpu_power_samples) if self.gpu_power_samples else 0.0
        avg_gpu_clock_graphics = sum(self.gpu_clock_graphics_samples) / len(self.gpu_clock_graphics_samples) if self.gpu_clock_graphics_samples else 0.0
        avg_gpu_clock_mem = sum(self.gpu_clock_mem_samples) / len(self.gpu_clock_mem_samples) if self.gpu_clock_mem_samples else 0.0
        
        return {
            "cpu_utilization": round(avg_cpu, 2),
            "gpu_utilization": round(avg_gpu, 2),
            "gpu_memory_utilization": round(avg_gpu_mem, 2),
            "gpu_temperature_c": round(avg_gpu_temp, 2),
            "gpu_power_watts": round(avg_gpu_power, 2),
            "gpu_graphics_clock_mhz": round(avg_gpu_clock_graphics, 2),
            "gpu_memory_clock_mhz": round(avg_gpu_clock_mem, 2)
        }

    def close(self):
        """Cleans up NVML resource handles."""
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
