import time
import os
import platform
import numpy as np

STUDENT_ID = 230110056

def get_cpu_model():
    try:
        if platform.system() == "Windows":
            import wmi
            return wmi.WMI().Win32_Processor()[0].Name.strip()
        elif platform.system() == "Darwin":
            import subprocess
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"

def main():
    os_info = f"{platform.system()} {platform.release()}"
    cpu_model = get_cpu_model()
    logical_cores = os.cpu_count()

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {os_info}")
    print(f"CPU Model: {cpu_model}")
    print(f"Logical Cores: {logical_cores}")

    np.random.seed(STUDENT_ID)
    data = np.random.rand(20_000_000).astype(np.float64)

    start_time = time.perf_counter()

    total_sum = 0.0
    for val in data:
        total_sum += val

    mean = total_sum / len(data)

    sum_sq_diff = 0.0
    for val in data:
        sum_sq_diff += (val - mean) ** 2

    std_dev = (sum_sq_diff / len(data)) ** 0.5

    end_time = time.perf_counter()
    execution_time_ms = (end_time - start_time) * 1000

    print(f"Sum: {total_sum:.6f}")
    print(f"Standard Deviation: {std_dev:.6f}")
    print(f"Execution Time: {execution_time_ms:.2f} ms")

if __name__ == "__main__":
    main()