from mpi4py import MPI
import numpy as np
# from numba import jit

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# @jit(nopython=True)
def bubble_sort_serial0(arr):
    # print(type(arr))
    # arr = list(arr)
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def bubble_sort_serial(arr, timer):
    if rank == 0:
        return bubble_sort_serial0(arr)
    else:
        return None


def bubble_sort(arr, timer):
    # Сортировка пузырьком
    # Сложность алгоритма: O(N^2)
    timer.start("sorter: Alexey [bubble_sort]")
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    timer.stop("sorter: Alexey [bubble_sort]")
    return arr


def sort(data, timer):
    timer.start("sorter: Alexey [scatter chunks]")
    sub_data = comm.scatter(np.array_split(data, size) if rank == 0 else None, root=0)
    timer.stop("sorter: Alexey [scatter chunks]")

    timer.start("sorter: Alexey [sorting local chunk]")
    local_sorted = bubble_sort(sub_data, timer)
    # local_sorted = bubble_sort(list(sub_data), timer)
    timer.stop("sorter: Alexey [sorting local chunk]")

    timer.start("sorter: Alexey [gather chunks]")
    gathered = comm.gather(local_sorted, root=0)
    timer.stop("sorter: Alexey [gather chunks]")

    if rank == 0:
        timer.start("sorter: Alexey [making merged]")
        merged = [item for sublist in gathered for item in sublist]
        timer.stop("sorter: Alexey [making merged]")
        timer.start("sorter: Alexey [sorting merged]")
        final_sorted = bubble_sort(merged, timer)
        timer.stop("sorter: Alexey [sorting merged]")
    else:
        final_sorted = None
    return final_sorted
