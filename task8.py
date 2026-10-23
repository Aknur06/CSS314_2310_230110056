import time
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230110056
MATRIX_SIZE = 1024

def matmul_non_tiled(A, B):
    return np.matmul(A, B)

def multiply_block(args):
    A, B, i_start, i_end, j_start, j_end, block_size = args
    N = A.shape[0]
    C_block = np.zeros((i_end - i_start, j_end - j_start), dtype=np.float64)
    
    for k in range(0, N, block_size):
        k_end = min(k + block_size, N)
        C_block += np.dot(A[i_start:i_end, k:k_end], B[k:k_end, j_start:j_end])
        
    return i_start, i_end, j_start, j_end, C_block

def matmul_tiled_parallel(A, B, block_size, num_workers):
    N = A.shape[0]
    C = np.zeros((N, N), dtype=np.float64)
    
    tasks = []
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        for j in range(0, N, block_size):
            j_end = min(j + block_size, N)
            tasks.append((A, B, i, i_end, j, j_end, block_size))
            
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(multiply_block, tasks)
        for i_start, i_end, j_start, j_end, C_block in results:
            C[i_start:i_end, j_start:j_end] = C_block
            
    return C

def main():
    max_cores = os.cpu_count() or 4
    np.random.seed(STUDENT_ID)
    
    A = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float64)
    B = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float64)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Matrix Dimension: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"Logical Cores: {max_cores}")
    print("=" * 65)
    print(f"{'Strategy / Geometry':<30} | {'Execution Time (ms)':<25}")
    print("=" * 65)

    start = time.perf_counter()
    _ = matmul_non_tiled(A, B)
    t_native = (time.perf_counter() - start) * 1000
    print(f"{'Native NumPy (Non-tiled BLAS)':<30} | {t_native:<25.2f}")

    start = time.perf_counter()
    _ = matmul_tiled_parallel(A, B, block_size=16, num_workers=max_cores)
    t_16 = (time.perf_counter() - start) * 1000
    print(f"{'Parallel Tiled (16x16)':<30} | {t_16:<25.2f}")

    start = time.perf_counter()
    _ = matmul_tiled_parallel(A, B, block_size=64, num_workers=max_cores)
    t_64 = (time.perf_counter() - start) * 1000
    print(f"{'Parallel Tiled (64x64)':<30} | {t_64:<25.2f}")

    print("=" * 65)
    print("CACHE EFFICIENCY & TILING ANALYSIS:")
    if t_64 < t_16:
        print("64x64 tiling outperformed 16x16 due to better L1/L2 cache line reuse and lower thread scheduling overhead.")
    else:
        print("16x16 tiling performed better due to smaller cache footprint per block.")
    print("=" * 65)

if __name__ == "__main__":
    main()