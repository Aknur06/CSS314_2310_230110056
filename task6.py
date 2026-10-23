import time
import os
from threading import Thread, Lock

STUDENT_ID = 230110056
INCREMENTS_PER_THREAD = 2_000_000
NUM_THREADS = 4
EXPECTED_VALUE = NUM_THREADS * INCREMENTS_PER_THREAD

class UnsafeCounter:
    def __init__(self):
        self.value = 0

    def increment(self):
        for _ in range(INCREMENTS_PER_THREAD):
            self.value += 1

class SafeCounter:
    def __init__(self):
        self.value = 0
        self.lock = Lock()

    def increment(self):
        for _ in range(INCREMENTS_PER_THREAD):
            with self.lock:
                self.value += 1

def run_threads(counter_obj):
    threads = []
    start_time = time.perf_counter()
    for _ in range(NUM_THREADS):
        t = Thread(target=counter_obj.increment)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    end_time = time.perf_counter()
    return (end_time - start_time) * 1000

def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Threads: {NUM_THREADS}")
    print(f"Increments per thread: {INCREMENTS_PER_THREAD:,}")
    print(f"Expected final value: {EXPECTED_VALUE:,}")
    print("=" * 60)

    unsafe_counter = UnsafeCounter()
    unsafe_time = run_threads(unsafe_counter)
    actual_unsafe = unsafe_counter.value
    lost_updates = EXPECTED_VALUE - actual_unsafe
    pct_lost = (lost_updates / EXPECTED_VALUE) * 100 if EXPECTED_VALUE > 0 else 0.0

    print("--- Unsynchronized (Race Condition) ---")
    print(f"Actual Value : {actual_unsafe:,}")
    print(f"Updates Lost : {lost_updates:,} ({pct_lost:.2f}%)")
    print(f"Time Taken   : {unsafe_time:.2f} ms")
    print("-" * 60)

    safe_counter = SafeCounter()
    safe_time = run_threads(safe_counter)
    actual_safe = safe_counter.value

    print("--- Synchronized (With Mutex Lock) ---")
    print(f"Actual Value : {actual_safe:,}")
    print(f"Time Taken   : {safe_time:.2f} ms")
    print("-" * 60)

    time_penalty = safe_time - unsafe_time
    slowdown = safe_time / unsafe_time if unsafe_time > 0 else 0.0

    print(f"Lock Overhead Penalty : +{time_penalty:.2f} ms ({slowdown:.2f}x slower)")
    print("=" * 60)

if __name__ == "__main__":
    main()