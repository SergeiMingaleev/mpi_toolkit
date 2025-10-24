#------------------------------------------------------------------
# Пример 7.1: Параллельное решение трёхдиагональной системы
#             линейных уравнений методом прогонки.
#
# Задача: найти вектор x, являющийся решением системы линейных
#         уравнений A*x = b с трёхдиагональной матрицей A,
#         максимально эффективно параллелизуя работу между
#         процессами с использованием MPI интерфейса.
#
# При параллелизации этого алгоритма мы будем считать, что общее
# число MPI процессов равно P.
# При этом процесс 0 будет занят одновременно и синхронизацией
# данных между процессами, и реальной работой - то есть, все P
# процессов будут "рабочими".
#
# Размер матрицы A равен (N, N), но ненулевыми являются только
# три главные диагонали, которые задаются тремя векторами:
# a (для верхней диагонали), b (для главной диагонали),
# и c (для нижней диагонали). Реально вектор b имеет N элементов,
# а векторы a и c оба имеют N-1 элементов - но для простоты
# мы будем выделять память в N элементов для каждого вектора.
# Считая, что N = P * K + L, где K и L - это целые числа,
# причём 0 <= L <= P-1, мы будем держать на каждом процессе
# либо по K+1 либо по К элементов каждого вектора, для
# максимальной балансировки памяти и вычислений по всем процессам.
#
# Точно так же будут разбиты на кусочки по процессам и
# векторы x и b (которые оба также состоят из N элементов).

# Реализация самого параллельного метода прогонки вынесена
# в функцию `parallel_tridiagonal_matrix_algorithm()` - остальная
# же часть кода занимается подготовкой и пересылкой данных
# между MPI процессами.
#
#------------------------------------------------------------------
# Этот пример (в его оригинальном виде) детально обсуждается
# в лекции Д.В. Лукьяненко "7. Параллельный вариант метода прогонки":
# https://youtu.be/CrqeCy-dZQI?list=PLcsjsqLLSfNCxGJjuYNZRzeDIFQDQ9WvC
#
# Данная программа объединяет в себе оба примера Д.В. Лукьяненко из
# его файлов "Example-7-1.py" (для вещественных чисел) и
# "Example-7-1-complex.py" (для комплексных чисел).
#------------------------------------------------------------------

import sys
from mpi4py import MPI
import numpy as np


is_complex_version = False

# Для удобства тестирования, пусть версия программы для комплексных
# чисел запускается при добавлении к программе аргумента 'complex':
# (mpiexec -n 4 python.exe Example-07-1.py complex)
if len(sys.argv) == 2 and sys.argv[1] == 'complex':
    is_complex_version = True

if is_complex_version:
    datatype = np.complex128
    MPI_datatype = MPI.DOUBLE_COMPLEX
else:
    datatype = np.float64
    MPI_datatype = MPI.DOUBLE


comm = MPI.COMM_WORLD
P = comm.Get_size()
rank = comm.Get_rank()


#------------------------------------------------------------------
def consecutive_tridiagonal_matrix_algorithm(a, b, c, d):
    """

    :param a:
    :param b:
    :param c:
    :param d:
    :return:
    """

    N = len(d)

    x = np.empty(N, dtype=datatype)

    for n in range(1, N):
        coef = a[n]/b[n-1]
        b[n] = b[n] - coef*c[n-1]
        d[n] = d[n] - coef*d[n-1]

    x[N-1] = d[N-1]/b[N-1]

    for n in range(N-2, -1, -1):
        x[n] = (d[n] - c[n]*x[n+1])/b[n]

    return x


def parallel_tridiagonal_matrix_algorithm(a_part, b_part, c_part, d_part):
    """

    :param a_part:
    :param b_part:
    :param c_part:
    :param d_part:
    :return:
    """
    N_part = len(d_part)

    for n in range(1, N_part):
        coef = a_part[n]/b_part[n-1]
        a_part[n] = -coef*a_part[n-1]
        b_part[n] = b_part[n] - coef*c_part[n-1]
        d_part[n] = d_part[n] - coef*d_part[n-1]

    for n in range(N_part-3, -1, -1):
        coef = c_part[n]/b_part[n+1]
        c_part[n] = -coef*c_part[n+1]
        a_part[n] = a_part[n] - coef*a_part[n+1]
        d_part[n] = d_part[n] - coef*d_part[n+1]

    if rank > 0:
        temp_array_send = np.array([a_part[0], b_part[0],
                                          c_part[0], d_part[0]], dtype=datatype)
    if rank < P-1:
        temp_array_recv = np.empty(4, dtype=datatype)

    if rank == 0:
        comm.Recv([temp_array_recv, 4, MPI_datatype],
                  source=1, tag=0, status=None)
    if rank in range(1, P-1):
        comm.Sendrecv(sendbuf=[temp_array_send, 4, MPI_datatype],
                      dest=rank-1, sendtag=0,
                      recvbuf=[temp_array_recv, 4, MPI_datatype],
                      source=rank+1, recvtag=MPI.ANY_TAG, status=None)
    if rank == P-1:
        comm.Send([temp_array_send, 4, MPI_datatype], dest=P-2, tag=0)

    if rank < P-1:
        coef = c_part[N_part-1] / temp_array_recv[1]
        b_part[N_part-1] = b_part[N_part-1] - coef*temp_array_recv[0]
        c_part[N_part-1] = - coef*temp_array_recv[2]
        d_part[N_part-1] = d_part[N_part-1] - coef*temp_array_recv[3]

    temp_array_send = np.array([a_part[N_part-1], b_part[N_part-1],
                             c_part[N_part-1], d_part[N_part-1]], dtype=datatype)

    if rank == 0:
        A_extended = np.empty((P, 4), dtype=datatype)
    else:
        A_extended = None

    comm.Gather([temp_array_send, 4, MPI_datatype],
                [A_extended, 4, MPI_datatype], root=0)

    if rank == 0:
        x_temp = consecutive_tridiagonal_matrix_algorithm(
            A_extended[:,0], A_extended[:,1], A_extended[:,2], A_extended[:,3])
    else:
        x_temp = None

    if rank == 0:
        rcounts_temp = np.empty(P, dtype=np.int32)
        displs_temp = np.empty(P, dtype=np.int32)
        rcounts_temp[0] = 1
        displs_temp[0] = 0
        for k in range(1, P):
            rcounts_temp[k] = 2
            displs_temp[k] = k - 1
    else:
        rcounts_temp = None
        displs_temp = None

    if rank == 0:
        x_part_last = np.empty(1, dtype=datatype)
        comm.Scatterv([x_temp, rcounts_temp, displs_temp, MPI_datatype],
                      [x_part_last, 1, MPI_datatype], root=0)
    else:
        x_part_last = np.empty(2, dtype=datatype)
        comm.Scatterv([x_temp, rcounts_temp, displs_temp, MPI_datatype],
                      [x_part_last, 2, MPI_datatype], root=0)

    x_part = np.empty(N_part, dtype=datatype)

    if rank == 0:
        for n in range(N_part-1):
            x_part[n] = (d_part[n] - c_part[n]*x_part_last[0])/b_part[n]
        x_part[N_part-1] = x_part_last[0]
    else:
        for n in range(N_part-1):
            x_part[n] = (d_part[n] - a_part[n]*x_part_last[0] -
                         c_part[n]*x_part_last[1]) / b_part[n]
        x_part[N_part-1] = x_part_last[1]

    return x_part


def diagonals_preparation(N_part):
    """
    Функция задает в качестве элементов диагоналей
    трёхдиагональной матрицы `A` произвольные числа.

    :param N_part: Число элементов в требуемых векторах.
    :return: Векторы диагоналей `a`, `b`, `c`.
    """
    a = np.empty(N_part, dtype=datatype)
    b = np.empty(N_part, dtype=datatype)
    c = np.empty(N_part, dtype=datatype)
    for n in range(N_part):
        if is_complex_version:
            a[n] = np.random.random_sample() + np.random.random_sample()*1j
            b[n] = np.random.random_sample() + np.random.random_sample()*1j
            c[n] = np.random.random_sample() + np.random.random_sample()*1j
        else:
            a[n] = np.random.random_sample()
            b[n] = np.random.random_sample()
            c[n] = np.random.random_sample()
    return a, b, c


# Определяем N - число элементов в модельном векторе `x`:
N = 10

if rank == 0:
    K, L = divmod(N, P)
    rcounts = np.empty(P, dtype=np.int32)
    displs = np.empty(P, dtype=np.int32)
    for m in range(0, P):
        if m < L:
            rcounts[m] = K + 1
        else:
            rcounts[m] = K
        if m == 0:
            displs[m] = 0
        else:
            displs[m] = displs[m - 1] + rcounts[m - 1]
else:
    rcounts = None
    displs = None

N_part = np.array(0, dtype=np.int32)
displ = np.array(0, dtype=np.int32)

comm.Scatter([rcounts, 1, MPI.INT],
             [N_part, 1, MPI.INT], root=0)
comm.Scatter([displs, 1, MPI.INT],
             [displ, 1, MPI.INT], root=0)

# Формируем на каждом MPI процессе свои кусочки диагоналей:
codiagonal_down_part, diagonal_part, codiagonal_up_part = diagonals_preparation(N_part)

# Задаём модельный вектор `x`, компонентами которого является
# последовательность натуральных чисел от 1 до N (включительно):
if rank == 0:
    x = np.array(range(1, N+1), dtype=np.float64)
else:
    x = np.empty(N, dtype=np.float64)

# Передаём модельный вектор `x` всем MPI процессам:
comm.Bcast([x, N, MPI.DOUBLE], root=0)

# Умножаем матрицу `А` на модельный вектор `x`.
# В результате получаем модельную правую часть,
# распределённую по всем MPI процессам по кусочкам:
b_part = np.zeros(N_part, dtype=datatype)
for n in range(N_part):
    if rank == 0 and n == 0:
        b_part[n] = (diagonal_part[n]*x[displ+n] +
                     codiagonal_up_part[n]*x[displ+n+1])
    elif rank == P-1 and n == N_part-1:
        b_part[n] = (codiagonal_down_part[n]*x[displ+n-1] +
                     diagonal_part[n]*x[displ+n])
    else:
        b_part[n] = (codiagonal_down_part[n]*x[displ+n-1] +
                     diagonal_part[n]*x[displ+n] +
                     codiagonal_up_part[n]*x[displ+n+1])

# Для сформированной матрицы `А` и модельной правой части `b`
# запускаем реализованный нами алгоритм мрешения СЛАУ
# с трёхдиагональной матрицей:
x_part = parallel_tridiagonal_matrix_algorithm(codiagonal_down_part,
                                               diagonal_part,
                                               codiagonal_up_part,
                                               b_part)

# Выводим результат и убеждаемся, что на каждом MPI процессе
# результат вычислений совпадает с кусочком модельного вектора:
print('For rank={}: x_part = {}'.format(rank, x_part))

#------------------------------------------------------------------
