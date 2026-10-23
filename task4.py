import time
import os
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230110056

def simulate_work(item):
    time.sleep(0.0001 * (item % 20))

def static_worker(items):
    start = time.perf_counter()
    for item in items:
        simulate_work(item)
    end = time.perf_counter()
    return end - start

def dynamic_worker(item):
    start = time.perf_counter()
    simulate_work(item)
    end = time.perf_counter()
    return start, end

def run_strategy_a(items, num_workers):
    chunk_size = len(items) // num_workers
    chunks = [items[i * chunk_size:(i + 1) * chunk_size] if i < num_workers - 1 else items[i * chunk_size:] for i in range(num_workers)]
    
    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        durations = list(executor.map(static_worker, chunks))
    end_total = time.perf_counter()
    
    total_time = (end_total - start_total) * 1000
    delta = (max(durations) - min(durations)) * 1000
    return total_time, delta

def run_strategy_b(items, num_workers, chunk_size):
    start_total = time.perf_counter()
    thread_finish_times = {}
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(dynamic_worker, items, chunksize=chunk_size))
        
    end_total = time.perf_counter()
    
    thread_durations = [r[1] - r[0] for r in results]
    total_time = (end_total - start_total) * 1000
    
    delta = (max(thread_durations) - min(thread_durations)) * 1000
    return total_time, delta

def main():
    num_workers = 4
    items = list(range(1000))
    
    print(f"Student ID: {STUDENT_ID}")
    print(f"Workers: {num_workers}")
    print(f"Items count: {len(items)}")
    print("-" * 60)
    
    time_a, delta_a = run_strategy_a(items, num_workers)
    time_b, delta_b = run_strategy_b(items, num_workers, chunk_size=5)
    
    print(f"{'Strategy':<30} | {'Total Time (ms)':<15} | {'Delta Min/Max (ms)':<18}")
    print("-" * 60)
    print(f"{'Strategy A (Static Chunk)':<30} | {time_a:<15.2f} | {delta_a:<18.2f}")
    print(f"{'Strategy B (Dynamic Chunk=5)':<30} | {time_b:<15.2f} | {delta_b:<18.2f}")
    print("-" * 60)

if __name__ == "__main__":
    main()