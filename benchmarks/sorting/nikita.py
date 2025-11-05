from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def quicksort(arr):
    arr = list(arr)
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def parallel_sort(data, comm, timer):
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    timer.start("sorter: Nikita [parallel_sort/array_split]")
    chunk = np.array_split(data, size)[rank]
    timer.stop("sorter: Nikita [parallel_sort/array_split]")

    timer.start("sorter: Nikita [parallel_sort/quicksort]")
    local_sorted = quicksort(chunk)
    timer.stop("sorter: Nikita [parallel_sort/quicksort]")

    timer.start("sorter: Nikita [parallel_sort/gather]")
    sorted_chunks = comm.gather(local_sorted, root=0)
    timer.stop("sorter: Nikita [parallel_sort/gather]")

    if rank == 0:
        timer.start("sorter: Nikita [parallel_sort/merge]")
        result = sorted_chunks[0]
        for part in sorted_chunks[1:]:
            result = merge(result, part)
        result = [int(x) for x in result]
        timer.stop("sorter: Nikita [parallel_sort/merge]")
        return result
    else:
        return None


def sort(data, timer):
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()

    timer.start("sorter: Nikita [bcast]")
    data = comm.bcast(data, root=0)
    timer.stop("sorter: Nikita [bcast]")

    comm.Barrier()

    timer.start("sorter: Nikita [parallel_sort]")
    sorted_data = parallel_sort(data, comm, timer)
    timer.stop("sorter: Nikita [parallel_sort]")

    comm.Barrier()

    return sorted_data
