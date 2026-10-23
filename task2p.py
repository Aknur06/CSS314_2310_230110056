import time
import math
import numpy as np
from threading import Thread, Lock

STUDENT_ID = 230110056
TOTAL_SAMPLES = 40_000_000
NUM_THREADS = 4
SAMPLES_PER_THREAD = TOTAL_SAMPLES // NUM_THREADS
CHUNK_SIZE = 100_000

class SharedPRNG:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
        self.lock = Lock()

    def get_points(self, count):
        with self.lock:
            return self.rng.uniform(0, 1, count), self.rng.uniform(0, 1, count)

def run_shared(shared_prng, count, results, index):
    iterations = count // CHUNK_SIZE
    inside = 0
    for _ in range(iterations):
        x, y = shared_prng.get_points(CHUNK_SIZE)
        inside += np.sum(x * x + y * y <= 1.0)
    results[index] = inside

def run_independent(thread_index, count, results):
    seed = (STUDENT_ID * 10) + thread_index
    rng = np.random.RandomState(seed)
    iterations = count // CHUNK_SIZE
    inside = 0
    for _ in range(iterations):
        x = rng.uniform(0, 1, CHUNK_SIZE)
        y = rng.uniform(0, 1, CHUNK_SIZE)
        inside += np.sum(x * x + y * y <= 1.0)
    results[thread_index] = inside

def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Total Monte Carlo Samples: {TOTAL_SAMPLES:,}")
    print(f"Threads: {NUM_THREADS}")
    print("=" * 65)

    shared_prng = SharedPRNG(STUDENT_ID)
    results_shared = [0] * NUM_THREADS
    threads_shared = []

    t0 = time.perf_counter()
    for i in range(NUM_THREADS):
        t = Thread(target=run_shared, args=(shared_prng, SAMPLES_PER_THREAD, results_shared, i))
        threads_shared.append(t)
        t.start()

    for t in threads_shared:
        t.join()
    t_shared = (time.perf_counter() - t0) * 1000

    pi_shared = 4.0 * sum(results_shared) / TOTAL_SAMPLES
    err_shared = abs(pi_shared - math.pi)

    print("--- Shared PRNG (Guarded by Mutex) ---")
    print(f"Computed Pi     : {pi_shared:.6f}")
    print(f"Absolute Error  : {err_shared:.6f}")
    print(f"Wall Time       : {t_shared:.2f} ms")
    print("-" * 65)

    results_indep = [0] * NUM_THREADS
    threads_indep = []

    t0 = time.perf_counter()
    for i in range(NUM_THREADS):
        t = Thread(target=run_independent, args=(i, SAMPLES_PER_THREAD, results_indep))
        threads_indep.append(t)
        t.start()

    for t in threads_indep:
        t.join()
    t_indep = (time.perf_counter() - t0) * 1000

    pi_indep = 4.0 * sum(results_indep) / TOTAL_SAMPLES
    err_indep = abs(pi_indep - math.pi)

    print("--- Independent PRNG (State Isolation) ---")
    print(f"Computed Pi     : {pi_indep:.6f}")
    print(f"Absolute Error  : {err_indep:.6f}")
    print(f"Wall Time       : {t_indep:.2f} ms")
    print("-" * 65)

    speedup = t_shared / t_indep if t_indep > 0 else 0.0
    print(f"Speedup Factor  : {speedup:.2f}x")
    print("=" * 65)

if __name__ == "__main__":
    main()