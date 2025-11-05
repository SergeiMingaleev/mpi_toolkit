from mpi4py import MPI
import numpy as np
import cpu_timer

import ivan
import alexey
import nikita
import george
import vadim
import hyperquicksort


def plot_data(data_random, data_sorted):
    # Строим график: индекс элемента -> значение
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(data_random, marker='.', linestyle='-', color='black', label='random')
    for key in data_sorted:
        plt.plot(data_sorted[key], marker='.', linestyle='-', label=key)
    plt.title("График отсортированного массива")
    plt.xlabel("Номер элемента (индекс)")
    plt.ylabel("Значение элемента")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()


def np_sort_quicksort(data, timer):
    # Быстрая сортировка, сортировка Хоара (англ. quicksort)
    # Алгоритм разработан английским информатиком Тони Хоаром
    # во время его работы в МГУ в 1960 году.
    # Один из самых быстрых известных универсальных алгоритмов
    # сортировки массивов: O(N * log[N])
    if rank == 0:
        return np.sort(data)
    else:
        return None


def np_sort_mergesort(data, timer):
    if rank == 0:
        return np.sort(data, kind='mergesort')
    else:
        return None


def np_sort_heapsort(data, timer):
    if rank == 0:
        return np.sort(data, kind='heapsort')
    else:
        return None


def np_sort_stable(data, timer):
    if rank == 0:
        return np.sort(data, kind='stable')
    else:
        return None


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

timer = cpu_timer.BenchmarkTimer()

sorters = {
    'np.sort(kind=quicksort)': np_sort_quicksort,
    'np.sort(kind=mergesort)': np_sort_mergesort,
    'np.sort(kind=heapsort)': np_sort_heapsort,
    'np.sort(kind=stable)': np_sort_stable,
    'Ivan': ivan.sort,
    'Alexey': alexey.sort,
    'Alexey_bubble_sort_serial': alexey.bubble_sort_serial,
    'Nikita': nikita.sort,
    'George': george.sort,
    'Vadim': vadim.parallel_merge_sort,
    'Vadim_iterative_merge_sort': vadim.iterative_merge_sort,
    # 'HyperQuickSort': hyperquicksort.sort,
    # 'HyperQuickSort_serial': hyperquicksort.sort_serial,
}

# Размер массива
# N = 100_000_000
# N = 10_000_000
# N = 1_000_000
# N = 100_000
# N = 10_000
# N = 1024
N = 100

if rank == 0:
    timer.start("Generating random array")
    # Генерация массива: значения от 0 до N
    np.random.seed(17)
    data_random = np.random.randint(low=0, high=N, size=N, dtype=np.int64)
    timer.stop("Generating random array")
else:
    data_random = None

data_sorted = {}
for key in sorters:
    data = data_random if data_random is None else data_random.copy()
    tag = f"sorter: {key}"
    timer.start(tag)
    timer.is_active = False
    data_sorted[key] = sorters[key](data, timer)
    timer.is_active = True
    timer.stop(tag)

if rank == 0:
    plot_data(data_random, data_sorted)
    timer.print_status()
