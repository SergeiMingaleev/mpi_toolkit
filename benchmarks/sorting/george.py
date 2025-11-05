from mpi4py import MPI
import numpy as np


def sort(a, timer):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        N = len(a)

    timer.start("sorter: George [bcast]")
    # --- Параллельная сортировка того же массива ---
    N = comm.bcast(N if rank == 0 else None, root=0)
    if rank != 0:
        a = np.empty(N, dtype=np.int64)
    comm.Bcast(a, root=0)
    timer.stop("sorter: George [bcast]")

    comm.Barrier()

    timer.start("sorter: George [array_split]")
    # Разрез и локальная сортировка
    chunks = np.array_split(a, size)
    timer.stop("sorter: George [array_split]")
    timer.start("sorter: George [local sort]")
    local = chunks[rank].copy()
    local.sort()
    timer.stop("sorter: George [local sort]")

    # Сэмплы и разделители (делаем всегда ровно P-1 разделителей)
    if len(local) > 0:
        # берём равномерные сэмплы (size-1 штук)
        smp_idx = np.linspace(0, len(local) - 1, num=size, dtype=int)[1:]
        samples = local[smp_idx]
    else:
        samples = np.empty(0, dtype=np.int64)

    all_samples = comm.gather(samples, root=0)
    if rank == 0:
        S = np.sort(np.concatenate(all_samples)) if len(all_samples) else np.empty(0, dtype=np.int64)
        if S.size > 0:
            # индексы для ровно (size-1) разделителей (работает даже если S.size < size-1)
            pick = np.linspace(0, max(S.size - 1, 0), num=size + 1, dtype=int)[1:-1]
            splitters = S[pick]
        else:
            splitters = np.full(size - 1, np.iinfo(np.int64).max, dtype=np.int64)
    else:
        splitters = None
    splitters = comm.bcast(splitters, root=0)

    # 3.3 Корзины и обмен (всегда size корзин)
    idx = np.searchsorted(local, splitters, side="right")
    bounds = np.concatenate(([0], idx, [len(local)]))
    buckets = [local[bounds[i]:bounds[i + 1]] for i in range(size)]

    received = comm.alltoall(buckets)

    # Финальная локальная сортировка
    recvbuf = np.concatenate(received) if received else np.empty(0, dtype=local.dtype)
    recvbuf.sort()

    # Сбор результата только для проверки и тайминга
    parts = comm.gather(recvbuf, root=0)
    comm.Barrier()
    # MPI.COMM_WORLD.Barrier()
    # MPI.Finalize()

    # --- Вывод и корректное завершение ---
    if rank == 0:
        par_sorted = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        return par_sorted
    else:
        return None

"""
if __name__ == "__main__":
    import sys
    try:
        main()
    finally:
        try:
            MPI.COMM_WORLD.Barrier()
        except Exception:
            pass
        try:
            MPI.Finalize()
        except Exception:
            pass
        sys.exit(0)
"""
