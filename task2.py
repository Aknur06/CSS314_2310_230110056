import time
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230110056

def parallel_sum(data, num_workers):
    chunk_size = len(data) // num_workers
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] if i < num_workers - 1 else data[i * chunk_size:] for i in range(num_workers)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(np.sum, chunks))
        
    return sum(results)

def main():
    max_cores = os.cpu_count() or 1
    worker_counts = sorted(list(set([1, 2, 4, max_cores, max_cores * 2])))
    
    print(f"Student ID: {STUDENT_ID}")
    print(f"Detected Logical Cores: {max_cores}")
    print("Generating 20,000,000 float64 elements...")
    
    np.random.seed(STUDENT_ID)
    data = np.random.rand(20_000_000).astype(np.float64)
    
    times = {}
    p_parallel = 0.95
    
    for p in worker_counts:
        parallel_sum(data[:1000], p)
        
        start = time.perf_counter()
        total_sum = parallel_sum(data, p)
        end = time.perf_counter()
        
        wall_time_ms = (end - start) * 1000
        times[p] = wall_time_ms

    t1 = times[1]
    
    print("\n" + "=" * 65)
    print(f"{'Cores':<8} | {'Wall Time (ms)':<16} | {'Speedup S(p)':<14} | {'Amdahl Upper Bound':<18}")
    print("=" * 65)
    
    for p in worker_counts:
        t_p = times[p]
        speedup = t1 / t_p
        amdahl = 1 / ((1 - p_parallel) + (p_parallel / p))
        
        print(f"{p:<8} | {t_p:<16.2f} | {speedup:<14.2f} | {amdahl:<18.2f}")
    
    print("=" * 65)

if __name__ == "__main__":
    main()