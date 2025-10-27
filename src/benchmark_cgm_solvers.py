# import os
# os.environ['OMP_NUM_THREADS'] = '1'

import warnings
import numpy as np
# import cupy as np

import mpi_toolkit as mpt

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
              1000, 2000] #, 3000, 4000]
    for M in M_list:
        yield M, M


# =============================================================================
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            message="overflow encountered in dot")

    mpt.config.numpy_lib = np

    verbose = False
    skip_init_time = False
    alpha = 0.0
    # alpha = 1.0e-13
    is_symmetric = False

    # solver = mpt.linalg.SolverSequential()
    # solver = mpt.linalg.SolverParallelBand1()
    # solver = mpt.linalg.SolverParallelBand2()
    # solver = mpt.linalg.SolverParallelBlock1()
    # solver = mpt.linalg.SolverParallelBlock2()
    solver = mpt.linalg.SolverParallelBlock3()

    mpt.print("="*80)
    mpt.print(f"SOLVER:  {type(solver)}")
    mpt.print(f"MATRIX:  matrix_TridiagThermal")
    # mpt.print("="*80)
    mpt.print(f"is_symmetric = {is_symmetric}")
    mpt.print(f"alpha = {alpha}")
    mpt.print(f"skip_init_time = {skip_init_time}")
    mpt.print("="*80)
    mpt.print("  N\t     M\t\t Time(sec)\t GFlops\t Error")

    # Цикл по размерам (M, N) матрицы A:
    for M, N in list_of_MN_case2():
        # Плохо обусловленная матрица:
        # A = mpt.linalg.matrix_Hilbert(M, N)

        # Хорошо (?) обусловленная матрица:
        A = mpt.linalg.matrix_TridiagThermal(M, N)

        # Создадим вектор `x` на процессе 0:
        t = mpt.arange(N) / N
        x0 = mpt.sin(2 * np.pi * t)

        # Создадим вектор `b` на процессе 0:
        b = mpt.dot(A, x0)

        # Начальное приближение для `x`:
        x = mpt.zeros(N, dtype=np.float64)

        #------------------------------------------------------------------
        # Решаем СЛАУ на всех процессах:
        x = solver.calc(A, b, x, is_symmetric, alpha, verbose, skip_init_time)

        # ------------------------------------------------------------------
        # Вывод результатов снова только на процессе 0:
        # Производительность расчётов:
        dt = solver.duration
        GFlops = solver.GFlops

        # Максимальная ошибка расчётов:
        error = mpt.error_max_abs(x0, x)

        mpt.print(f"{N:5}\t {M:7}\t {dt:.6f}\t {GFlops:.3f}\t {error:.5g}")

        # Для контроля, напечатаем найденное решение на консоли:
        # mpt.print(f"\nFinal solution:\n   x = {x}")
        # mpt.print(f"Error is:")
        # mpt.print(f"    dx = {x - x0}")
        # mpt.print(f"   |dx|^2 = {np.dot(x0 - x, x0 - x)}")
        # mpt.print(f"   |x0|^2 = {np.dot(x0, x0)}")
        # mpt.print(f"    |x|^2 = {np.dot(x, x)}")
        # plot_solution(x)
    mpt.print("="*80)

# =============================================================================
