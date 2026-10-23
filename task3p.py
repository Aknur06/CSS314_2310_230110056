import time
import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230110056
ARRAY_SIZE = 2_000_000

def local_scan_worker(args):
    chunk, offset = args
    scanned = np.cumsum(chunk)
    if offset != 0:
        scanned += offset
    return scanned

def main():
    max_cores = os.cpu_count() or 4

    np.random.seed(STUDENT_ID)
    input_array = np.random.randint(1, 11, size=ARRAY_SIZE, dtype=np.int64)

    t0 = time.perf_counter()
    serial_output = np.cumsum(input_array)
    serial_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    chunk_size = ARRAY_SIZE // max_cores
    chunks = [
        input_array[i * chunk_size : (i + 1) * chunk_size]
        if i < max_cores - 1
        else input_array[i * chunk_size :]
        for i in range(max_cores)
    ]

    chunk_sums = [np.sum(c, dtype=np.int64) for c in chunks]

    offsets = np.zeros(max_cores, dtype=np.int64)
    offsets[1:] = np.cumsum(chunk_sums[:-1])

    tasks = [(chunks[i], offsets[i]) for i in range(max_cores)]

    with ThreadPoolExecutor(max_workers=max_cores) as executor:
        scanned_chunks = list(executor.map(local_scan_worker, tasks))

    parallel_output = np.concatenate(scanned_chunks)
    parallel_time = (time.perf_counter() - t0) * 1000

    try:
        assert parallel_output[-1] == serial_output[-1], "Final element mismatch!"
        assert np.array_equal(parallel_output, serial_output), "Array match failed!"
        verification_status = "PASS"
    except AssertionError as e:
        verification_status = "FAIL"
        print(f"VERIFICATION ERROR: {e}")
        sys.exit(1)

    speedup = serial_time / parallel_time if parallel_time > 0 else 0.0

    print(f"Student ID: {STUDENT_ID}")
    print(f"Array Size: {ARRAY_SIZE:,}")
    print(f"Logical Cores: {max_cores}")
    print("=" * 65)
    print(f"Serial Time         : {serial_time:.2f} ms")
    print(f"Parallel Time       : {parallel_time:.2f} ms")
    print(f"Speedup             : {speedup:.2f}x")
    print(f"Verification Status : {verification_status}")
    print("=" * 65)

if __name__ == "__main__":
    main()