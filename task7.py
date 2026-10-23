import time
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230110056
N = 30_000_000

def kernel_a_worker(chunk):
    return np.copy(chunk)

def kernel_b_worker(chunk):
    return np.sin(chunk) * np.cos(chunk) + np.tan(chunk)

def run_benchmark(kernel_func, data, num_workers):
    chunk_size = len(data) // num_workers
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] if i < num_workers - 1 else data[i * chunk_size:] for i in range(num_workers)]
    
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(kernel_func, chunks))
    np.concatenate(results)
    end = time.perf_counter()
    
    return (end - start) * 1000

def main():
    max_cores = os.cpu_count() or 4
    cores_to_test = [1, 2, max_cores]
    
    np.random.seed(STUDENT_ID)
    A = np.random.rand(N).astype(np.float64)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Array Size (N): {N:,}")
    print(f"Max Logical Cores: {max_cores}")
    print("=" * 70)

    print("\n--- Kernel A: Memory-Bound (Vector Copy: B[i] = A[i]) ---")
    t1_a = run_benchmark(kernel_a_worker, A, 1)
    eff_a_max = 0.0
    for c in cores_to_test:
        t_c = run_benchmark(kernel_a_worker, A, c)
        speedup = t1_a / t_c
        eff = (speedup / c) * 100
        if c == max_cores:
            eff_a_max = eff
        print(f"Cores: {c:<2} | Time: {t_c:>8.2f} ms | Speedup: {speedup:>6.2f}x | Efficiency: {eff:>6.2f}%")

    print("\n--- Kernel B: Compute-Bound (Trig Ops: sin*cos + tan) ---")
    t1_b = run_benchmark(kernel_b_worker, A, 1)
    eff_b_max = 0.0
    for c in cores_to_test:
        t_c = run_benchmark(kernel_b_worker, A, c)
        speedup = t1_b / t_c
        eff = (speedup / c) * 100
        if c == max_cores:
            eff_b_max = eff
        print(f"Cores: {c:<2} | Time: {t_c:>8.2f} ms | Speedup: {speedup:>6.2f}x | Efficiency: {eff:>6.2f}%")

    print("\n" + "=" * 70)
    print("ANALYSIS:")
    if eff_a_max < eff_b_max:
        print("Kernel A (Memory-Bound Vector Copy) stalled in scaling due to memory channel saturation.")
    else:
        print("Kernel B experienced greater scaling overhead.")
    print("=" * 70)

if __name__ == "__main__":
    main()