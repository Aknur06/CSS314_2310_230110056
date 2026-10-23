import time
import os
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230110056
ITERATIONS = 50_000_000

def unpadded_worker(shared_counts, thread_id):
    for _ in range(ITERATIONS):
        shared_counts[thread_id] += 1

def local_worker(shared_counts, thread_id):
    local_count = 0
    for _ in range(ITERATIONS):
        local_count += 1
    shared_counts[thread_id] = local_count

def run_unpadded(num_threads):
    shared_counts = [0] * num_threads
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(unpadded_worker, shared_counts, i) for i in range(num_threads)]
        for f in futures:
            f.result()
    end = time.perf_counter()
    return (end - start) * 1000

def run_local(num_threads):
    shared_counts = [0] * num_threads
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(local_worker, shared_counts, i) for i in range(num_threads)]
        for f in futures:
            f.result()
    end = time.perf_counter()
    return (end - start) * 1000

def main():
    num_threads = os.cpu_count() or 4

    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical/Logical Cores used: {num_threads}")
    print(f"Iterations per thread: {ITERATIONS:,}")
    print("-" * 50)

    unpadded_time = run_unpadded(num_threads)
    local_time = run_local(num_threads)

    ratio = unpadded_time / local_time if local_time > 0 else 0.0

    print(f"Unpadded Shared Time : {unpadded_time:.2f} ms")
    print(f"Padded/Local Time    : {local_time:.2f} ms")
    print(f"Execution Time Ratio : {ratio:.2f}x")
    print("-" * 50)

if __name__ == "__main__":
    main()