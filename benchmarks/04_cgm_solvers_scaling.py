# import os
# os.environ['OMP_NUM_THREADS'] = '1'

import warnings
import numpy as np
# import cupy as np
from mpi4py import MPI

from linalg_solvers import matrix_TridiagThermal, matrix_Hilbert
from linalg_solvers import SolverSequential
from linalg_solvers import SolverParallelBand1
from linalg_solvers import SolverParallelBand2
from linalg_solvers import SolverParallelBlock1

# =============================================================================
def plot_solution(x):
    # Подготовим рисунок и данные для него:
    import matplotlib.pyplot as plt

    plt.style.use('dark_background')
    fig = plt.figure()
    ax = plt.axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i')
    ax.set_ylabel('x[i]')

    # индексы элементов вектора `x`:
    ii = np.arange(np.int32(N))

    # Нарисуем полученное нами выше полное решение:
    ax.plot(ii, x, '-y', lw=3, label='conj_grad (A)')
    # Для `cupy` нужно добавлять к массивам метод .get(),
    # чтобы перевести их сначала в `numpy` массивы на CPU:
    # ax.plot(ii.get(), x.get(), '-y', lw=3, label='conj_grad (A)')

    plt.legend()
    plt.show()


# =============================================================================
def list_of_MN_case1():
    M_list = [200, 400, 600, 800, 1000, 1500, 2000, 2500,
         5000, 10_000, 20_000, 50_000, 100_000,
         150_000, 200_000, 250_000, 500_000,
         1_000_000, 2_000_000]
    N = 100
    for M in M_list:
        yield M, N


# =============================================================================
def list_of_MN_case2():
    M_list = [100, 200, 300, 400, 500,
              1000, 2000, 3000]
    for M in M_list:
        yield M, M


# =============================================================================
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            message="overflow encountered in dot")

    # Работаем с коммуникатором по всем доступным процессам:
    comm = MPI.COMM_WORLD
    P = comm.Get_size()
    rank = comm.Get_rank()

    # Заглушки для всех процессов (кроме процесса 0 ниже):
    A = b = x = x0 = None

    if rank == 0:
        print("  N\t     M\t\t Time(sec)\t GFlops\t Error")

    # Цикл по размерам (M, N) матрицы A:
    for M, N in list_of_MN_case2():
        # Создадим нужные нам переменные только на процессе 0:
        if rank == 0:
            # Случайная матрица обычно очень-очень плохо обусловлена -
            # мы больше не будем использовать её в таких тестах:
            # np.random.seed(71)
            # A = np.random.rand(M, N)

            # Плохо обусловленная матрица:
            # A = matrix_Hilbert(M, N, numpy_lib=np)

            # Хорошо (?) обусловленная матрица:
            A = matrix_TridiagThermal(M, N, numpy_lib=np)

            # Создадим вектор `x` на процессе 0:
            t = np.arange(N) / N
            x0 = np.sin(2 * np.pi * t)

            # Создадим вектор `b` на процессе 0:
            b = np.dot(A, x0)

            # Начальное приближение для `x`:
            x = np.zeros(N, dtype=np.float64)

        #------------------------------------------------------------------
        # Решаем СЛАУ на всех процессах:
        verbose = False
        skip_init_time = True
        alpha = 0.0
        alpha = 1.0e-12
        is_symmetric = False

        # solver = SolverSequential(numpy_lib=np)
        # solver = SolverParallelBand1(numpy_lib=np)
        # solver = SolverParallelBand2(numpy_lib=np)
        solver = SolverParallelBlock1(numpy_lib=np)

        x = solver.calc(A, b, x, is_symmetric, alpha, verbose, skip_init_time)

        # ------------------------------------------------------------------
        # Вывод результатов снова только на процессе 0:
        if rank == 0:
            # Производительность расчётов:
            dt = solver.duration
            GFlops = solver.GFlops

            # Максимальная ошибка расчётов:
            error = np.max(np.abs(x0 - x))

            print(f"{N:5}\t {M:7}\t {dt:.6f}\t {GFlops:.3f}\t {error:.5g}")

            # Для контроля, напечатаем найденное решение на консоли:
            # print(f"\nFinal solution:\n   x = {x}")
            # print(f"Error is:")
            # print(f"    dx = {x - x0}")
            # print(f"   |dx|^2 = {np.dot(x0 - x, x0 - x)}")
            # print(f"   |x0|^2 = {np.dot(x0, x0)}")
            # print(f"    |x|^2 = {np.dot(x, x)}")
            # plot_solution(x)

# =============================================================================
