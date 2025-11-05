import numpy as np
import heapq
from mpi4py import MPI
# параллельно и последовательно сортируем методом слияния

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# слияние с использованием heapq
def merge(left, right):
    if len(left) == 0:
        return right
    if len(right) == 0:
        return left
    return np.array(list(heapq.merge(left, right)))


# сортировка для последовательной версии
def iterative_merge_sort(arr, timer):
    if arr is None:
        return None
    arr = arr.copy()
    n = len(arr)
    curr_size = 1
    while curr_size < n:
        for left_start in range(0, n, 2 * curr_size):
            mid = min(left_start + curr_size, n)
            right_end = min(left_start + 2 * curr_size, n)
            if mid < right_end:
                merged = merge(arr[left_start:mid], arr[mid:right_end])
                arr[left_start:right_end] = merged
        curr_size *= 2
    return arr


# параллельная сортировка с MPI
def parallel_merge_sort(arr, timer):
    if rank == 0:
        chunk_size = len(arr) // size
        ostatok = len(arr) % size
        sendcounts = [chunk_size + 1 if i < ostatok else chunk_size for i in range(size)]
        displs = np.cumsum([0] + sendcounts[:-1])
    else:
        sendcounts = None
        displs = None
    sendcounts = comm.bcast(sendcounts, root=0)

    local_arr = np.empty(sendcounts[rank], dtype=arr.dtype if rank == 0 else np.int64)
    # Распределение данных
    if rank == 0:
        comm.Scatterv([arr, sendcounts, displs, MPI.LONG_LONG], local_arr, root=0)
    else:
        comm.Scatterv(None, local_arr, root=0)
    local_sorted = iterative_merge_sort(local_arr, timer)
    # слияние отсортированных частей
    step = 1
    while step < size:
        if rank % (2 * step) == 0:
            partner = rank + step
            # если существует партнер
            if partner < size:
                recv_size = comm.recv(source=partner, tag=step)
                if recv_size > 0:
                    received = np.empty(recv_size, dtype=local_sorted.dtype)
                    comm.Recv(received, source=partner, tag=step + 1)
                    local_sorted = merge(local_sorted, received)
        elif rank % (2 * step) == step:
            partner = rank - step

            comm.send(len(local_sorted), dest=partner, tag=step)
            if len(local_sorted) > 0:
                comm.Send(local_sorted, dest=partner, tag=step + 1)
            local_sorted = np.array([], dtype=local_sorted.dtype)
        step *= 2

    return local_sorted if rank == 0 else None
