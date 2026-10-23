import time
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230110056
NUM_TASKS = 10_000
DATA = b"x" * 1024

def hash_task(_):
    val = DATA
    for _ in range(500):
        val = hashlib.sha256(val).digest()
    return val

def main():
    cores = os.cpu_count() or 4
    raw_sizes = [1, max(1, cores // 2), cores, cores * 2, cores * 4, cores * 16, 256]
    
    pool_sizes = []
    for sz in raw_sizes:
        if sz not in pool_sizes:
            pool_sizes.append(sz)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical Cores: {cores}")
    print(f"Total Tasks: {NUM_TASKS:,}")
    print("=" * 75)
    print(f"{'Pool Size':<12} | {'Time (ms)':<15} | {'Speedup':<12} | {'Status':<25}")
    print("=" * 75)

    results = []
    t1 = 0.0

    for p in pool_sizes:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=p) as executor:
            list(executor.map(hash_task, range(NUM_TASKS)))
        elapsed = (time.perf_counter() - t0) * 1000

        if p == 1:
            t1 = elapsed
            speedup = 1.00
        else:
            speedup = t1 / elapsed if elapsed > 0 else 0.0

        results.append((p, elapsed, speedup))

    min_time = float("inf")
    best_pool = cores

    for p, elapsed, _ in results:
        if elapsed < min_time:
            min_time = elapsed
            best_pool = p

    for p, elapsed, speedup in results:
        if p == best_pool:
            status = "OPTIMAL"
        elif p > best_pool and elapsed > min_time * 1.02:
            status = "DEGRADED (Context Switch)"
        elif p < best_pool:
            status = "UNDERUTILIZED"
        else:
            status = "STABLE"

        print(f"{p:<12} | {elapsed:>12.2f} ms | {speedup:>10.2f}x | {status:<25}")

    print("=" * 75)

if __name__ == "__main__":
    main()