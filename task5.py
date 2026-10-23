import time
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230110056

def serial_vector_ops(x, a, b):
    return a * x + b

def parallel_worker(chunk, a, b):
    return a * chunk + b

def parallel_vector_ops(x, a, b, num_workers):
    chunk_size = len(x) // num_workers
    if chunk_size == 0:
        return serial_vector_ops(x, a, b)
    
    chunks = [x[i * chunk_size:(i + 1) * chunk_size] if i < num_workers - 1 else x[i * chunk_size:] for i in range(num_workers)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(lambda c: parallel_worker(c, a, b), chunks))
        
    return np.concatenate(results)

def main():
    max_cores = os.cpu_count() or 4
    sizes = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    a, b = 2.5, 1.5

    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical Cores: {max_cores}")
    print("=" * 65)
    print(f"{'N':<12} | {'Serial (ms)':<15} | {'Parallel (ms)':<15} | {'Winner':<10}")
    print("=" * 65)

    crossover_found = None

    for N in sizes:
        np.random.seed(STUDENT_ID)
        x = np.random.rand(N).astype(np.float64)

        serial_vector_ops(x[:10], a, b)
        start = time.perf_counter()
        res_serial = serial_vector_ops(x, a, b)
        t_serial = (time.perf_counter() - start) * 1000

        parallel_vector_ops(x[:10], a, b, max_cores)
        start = time.perf_counter()
        res_parallel = parallel_vector_ops(x, a, b, max_cores)
        t_parallel = (time.perf_counter() - start) * 1000

        winner = "Parallel" if t_parallel < t_serial else "Serial"

        print(f"{N:<12,} | {t_serial:<15.4f} | {t_parallel:<15.4f} | {winner:<10}")

        if winner == "Parallel" and crossover_found is None:
            crossover_found = N

    print("=" * 65)
    if crossover_found:
        idx = sizes.index(crossover_found)
        prev_n = sizes[idx - 1] if idx > 0 else 0
        print(f"Crossover Point: Multi-threading becomes faster between N = {prev_n:,} and N = {crossover_found:,}")
    else:
        print("Parallel execution remained slower across all tested N sizes.")

if __name__ == "__main__":
    main()