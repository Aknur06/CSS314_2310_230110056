import time
import numpy as np
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory

STUDENT_ID = 230110056
SIZES = [100_000, 1_000_000, 10_000_000]

def path_a_worker(q):
    _ = q.get()

def path_b_worker(shm_name, shape, dtype):
    existing_shm = SharedMemory(name=shm_name)
    data = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    _ = data[0] + data[-1]
    existing_shm.close()

def benchmark_path_a(n):
    data = np.ones(n, dtype=np.float64)
    q = Queue()
    p = Process(target=path_a_worker, args=(q,))
    t0 = time.perf_counter()
    p.start()
    q.put(data)
    p.join()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000

def benchmark_path_b(n):
    data = np.ones(n, dtype=np.float64)
    shm = SharedMemory(create=True, size=data.nbytes)
    shm_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    
    t0 = time.perf_counter()
    shm_array[:] = data[:]
    p = Process(target=path_b_worker, args=(shm.name, data.shape, data.dtype))
    p.start()
    p.join()
    t1 = time.perf_counter()
    
    shm.close()
    shm.unlink()
    return (t1 - t0) * 1000

def main():
    print(f"Student ID: {STUDENT_ID}")
    print("=" * 70)
    print(f"{'Array Size (N)':<15} | {'Path A (IPC Queue)':<22} | {'Path B (SharedMemory)':<22}")
    print("=" * 70)

    for n in SIZES:
        t_a = benchmark_path_a(n)
        t_b = benchmark_path_b(n)
        print(f"{n:<15,} | {t_a:>18.2f} ms | {t_b:>18.2f} ms")

    print("=" * 70)

if __name__ == "__main__":
    main()