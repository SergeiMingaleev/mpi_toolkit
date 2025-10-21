#------------------------------------------------------------------
# Проверка скорости работы библиотеки `numpy` на одном
# компьютере. Практически все современные сборки `numpy`
# уже используют параллелизацию выполнения тех или иных
# операций `numpy` в многоядерных системах, и для правильной
# параллелизации своего кода важно понимать вашу ситуацию.
#
# https://superfastpython.com/numpy-multithreaded-parallelism/
#------------------------------------------------------------------

import os

# Установим ограничение на число потоков,
# используемых для векторных операций внутри
# библиотеки `numpy`.
# 1) это нужно делать до импорта самой `numpy`.
# 2) разные сборки `numpy` используют разные
#    реализации библиотек Blas и Lapack, и
#    распараллеливание в них управляется разными
#    переменными окружения - меняем их все для
#    надёжности.

N_THREADS = '1'
#N_THREADS = '8'

os.environ['OMP_NUM_THREADS'] = N_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = N_THREADS
os.environ['MKL_NUM_THREADS'] = N_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = N_THREADS

import numpy as np
# import cupy as np
from linalg_solvers import BenchmarkTimer


# Размер матриц будет NxN:
N = 3500

# Сделаем две случайные матрицы с размером NxN:
np.random.seed(17)
data1 = np.random.rand(N, N)
data2 = np.random.rand(N, N)

# Проверим скорость перемножения матриц:

timer = BenchmarkTimer()

timer.start('dot')
result = data1.dot(data2)
timer.stop('dot')

timer.print_status()
#------------------------------------------------------------------
