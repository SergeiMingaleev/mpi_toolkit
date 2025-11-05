from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
P = comm.Get_size()



# convert decimal x to binary array bin
def binary(x: int, dim: int):
    bin = np.zeros(dim, dtype=np.int64)
    i = dim - 1  # start with least signficant position
    while x > 0:
        bin[i] = x % 2
        x //= 2
        i -= 1   # go to next most significant position
    return bin


# check which processes generate pivot
def check_id(bin_rank, step, dim):
    for i in range(dim-1, step-1, -1):
        if bin_rank[i] == 1:
            return False
    return True


# partition array around the pivot
def hyper_partition(array, low, high, pivot):
    i = low - 1
    for j in range(low, high):
        if array[j] <= pivot:
            i += 1
            array[j], array[i] = array[i], array[j]
    array[high], array[i+1] = array[i+1], array[high]
    return i if (array[i+1] > pivot) else (i+1)


# partition array for serial quicksort
def partition(array, low, high):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j] <= pivot:
            i += 1
            array[j], array[i] = array[i], array[j]
    array[high], array[i+1] = array[i+1], array[high]
    return i + 1


# serial quicksort
def quicksort(array, low, high):
    if low < high:
        pi = partition(array, low, high)
        quicksort(array, low, pi-1)
        quicksort(array, pi+1, high)


def sort_serial(arr, timer):
    if rank != 0:
        return None
    quicksort(arr, 0, len(arr)-1)
    return arr


def sort(arr, timer):
    if rank == 0:
        N = len(arr)
        # declare sorted array
        sorted_array = np.empty(N, dtype=np.int64)
    N = comm.bcast(N if rank == 0 else None, root=0)
    if rank != 0:
        arr = sorted_array = None

    dim = int(np.log2(P))         # dimension of hypercube
    chunk_size = N // P           # size of chuck array of each process

    # chunk array of each process
    chunk_array = np.empty(chunk_size, dtype=np.int64)

    # binary process id
    bin_rank = binary(rank, dim)

    comm.Scatter([arr, chunk_size, MPI.LONG_LONG],
                 [chunk_array, chunk_size, MPI.LONG_LONG], root=0)

    for i in range(dim):
        # define color of sub_hypercubes based on (dim-1)-hypercube partition
        color = rank // (P // (i+1))

        # Communicator for SUB_HYPERCUBE:
        sub_comm = comm.Split(color, rank)

        # process id in sub_hypercubes
        sub_rank = sub_comm.Get_rank()

        # number of processes in sub_hypercubes
        sub_P = sub_comm.Get_size()

        if check_id(bin_rank, i, dim):
            # pick last element of array as pivot
            pivot = np.array(chunk_array[chunk_size-1], dtype=np.int64)
        else:
            pivot = np.array(0, dtype=np.int64)

        sub_comm.Bcast([pivot, 1, MPI.LONG_LONG], root=0)

        # partition array around pivot and return pivot position
        # chunk_array = chunk_array.copy()?
        position = hyper_partition(chunk_array, 0, chunk_size-1, pivot)

        if sub_rank // (sub_P // 2) == 1:
            # if process in upper half of sub_hypercube
            # send first elements of array up to the pivot position
            send_array = chunk_array[:position+1]
            # but keep the others
            keep_array = chunk_array[position+1:]
        else:
            # otherwise send the remaining elements of array including the pivot
            send_array = chunk_array[position+1:]
            # but keep the others
            keep_array = chunk_array[:position+1]

        # partner process in sub_hypercube
        if sub_rank // (sub_P // 2) == 0:
            # determine partner process in the upper half of the sub_hypercube
            partner_process = sub_rank + sub_P // 2
        else:
            # determine partner process in the lower half of the sub_hypercube
            partner_process = sub_rank - sub_P // 2

        # size of array to be sent (in int format)
        send_array_size = np.array(len(send_array), dtype=np.int64)

        # send size of array to be sent
        sub_comm.Send([send_array_size, 1, MPI.LONG_LONG],
                      dest=partner_process, tag=dim)

        # receive size of array to be received
        receive_array_size = np.array(0, dtype=np.int64)
        sub_comm.Recv([receive_array_size, 1, MPI.LONG_LONG],
                      source=partner_process, tag=dim, status=None)

        # array to be received
        receive_array = np.empty(receive_array_size, dtype=np.int64)

        # send array
        sub_comm.Send([send_array, send_array_size, MPI.LONG_LONG],
                      dest=partner_process, tag=dim)

        # receive array
        sub_comm.Recv([receive_array, receive_array_size, MPI.LONG_LONG],
                      source=partner_process, tag=dim, status=None)

        # redefine chunk_size
        chunk_size = len(keep_array) + receive_array_size
        # clear chunk_array array
        chunk_array = np.empty(chunk_size, dtype=np.int64)

        # repopulate first positions of chunk_array with keep_array
        chunk_array[:len(keep_array)] = keep_array[:]
        # repopulate remaining positions with receive_array
        chunk_array[len(keep_array):] = receive_array[:]

        sub_comm.Free()

    quicksort(chunk_array, 0, chunk_size-1)
    # chunk_array.sort()
    # chunk_array = np.sort(chunk_array)

    # array for chunk_array (in array format)
    sorted_chunk_array = np.empty(chunk_size, dtype=np.int64)
    # array with size of each chunk_array
    count = np.empty(P, dtype=np.int64)
    # array with placement index of each chunk_array
    displacement = np.empty(P, dtype=np.int64)

    # sorted chunk_array (in array format)
    sorted_chunk_array[:] = chunk_array[:]

    if rank==0:
        count[0] = chunk_size
        displacement[0] = 0

        # receive the size of each processor's chunk_array on processor 0
        count_recv = np.array(0, dtype=np.int64)
        for i in range(1, P):
            comm.Recv([count_recv, 1, MPI.LONG_LONG], source=i, tag=i, status=None)
            count[i] = np.int64(count_recv)

        temp = 0
        for i in range(1, P):
            temp += count[i-1]
            # increment the placement index of each chunk_array
            displacement[i] = temp
    else:
        # send the size of chunk_array to processor 0
        comm.Send([chunk_size, 1, MPI.LONG_LONG], dest=0, tag=rank)

    # comm.Gatherv([sorted_chunk_array, chunk_size, MPI.LONG_LONG],
    #              [sorted_array, count, displacement, MPI.LONG_LONG], root=0)

    if rank==0:
       comm.Gatherv([sorted_chunk_array, chunk_size, MPI.LONG_LONG],
                    [sorted_array, count, displacement, MPI.LONG_LONG], root=0)
    else:
       comm.Gatherv([sorted_chunk_array, chunk_size, MPI.LONG_LONG],
                    [None, None, None, MPI.LONG_LONG], root=0)

    return sorted_array
