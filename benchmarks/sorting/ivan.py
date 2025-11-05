from mpi4py import MPI
import numpy as np
import heapq

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def sort(data, timer):
    if rank == 0:
        timer.start("sorter: Ivan [array_split]")
        # Делим массив на части для процессов
        chunks = np.array_split(data, size)
        timer.stop("sorter: Ivan [array_split]")
    else:
        chunks = None

    # Рассылаем части массива процессам
    timer.start("sorter: Ivan [scatter chunks]")
    local_data = comm.scatter(chunks, root=0)
    timer.stop("sorter: Ivan [scatter chunks]")

    # Каждый процесс сортирует свою часть
    timer.start("sorter: Ivan [sorting local chunk]")
    local_sorted = np.sort(local_data)
    timer.stop(f"sorter: Ivan [sorting local chunk]")

    # Собираем отсортированные куски обратно на root
    timer.start("sorter: Ivan [gather chunks]")
    gathered = comm.gather(local_sorted, root=0)
    timer.stop("sorter: Ivan [gather chunks]")

    if rank == 0:
        timer.start("sorter: Ivan [merging chunks]")
        # Сливаем отсортированные куски в один массив
        # TODO: неэффективно!
        final_sorted = list(heapq.merge(*gathered))
        # final_sorted = np.array(heapq.merge(*gathered))
        timer.stop("sorter: Ivan [merging chunks]")
    else:
        final_sorted = None
    return final_sorted
