import time
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230110056
N = 8000

def mode_a_worker(row_chunk):
    return np.sum(row_chunk)

def mode_b_worker(col_chunk):
    total = 0.0
    for col in range(col_chunk.shape[1]):
        total += np.sum(col_chunk[:, col])
    return total

def main():
    max_cores = os.cpu_count() or 4

    np.random.seed(STUDENT_ID)
    matrix = np.random.rand(N, N).astype(np.float64)

    chunk_size = N // max_cores
    row_chunks = [
        matrix[i * chunk_size : (i + 1) * chunk_size, :]
        if i < max_cores - 1
        else matrix[i * chunk_size :, :]
        for i in range(max_cores)
    ]
    col_chunks = [
        matrix[:, i * chunk_size : (i + 1) * chunk_size]
        if i < max_cores - 1
        else matrix[:, i * chunk_size :]
        for i in range(max_cores)
    ]

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_cores) as executor:
        sum_a = sum(executor.map(mode_a_worker, row_chunks))
    t_a = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_cores) as executor:
        sum_b = sum(executor.map(mode_b_worker, col_chunks))
    t_b = (time.perf_counter() - start) * 1000

    penalty_factor = t_b / t_a if t_a > 0 else 0.0

    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical Cores: {max_cores}")
    print(f"Matrix Size: {N}x{N} (float64)")
    print("=" * 60)
    print(f"Mode A (Row-Major) Runtime    : {t_a:.2f} ms")
    print(f"Mode B (Column-Major) Runtime : {t_b:.2f} ms")
    print(f"Cache Penalty Factor          : {penalty_factor:.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    main()