#------------------------------------------------------------------
# Пример 2.2: Параллельное решение системы линейных уравнений
#             A*x = b
#
# Задача: найти вектор x, являющийся решением системы линейных
#         уравнений A*x = b

# Размер матрицы A равен (M, N) - M строк и N колонок.
# Эта матрица будет в полном виде зачитана (и храниться в массиве A)
# на процессе 0 (точно как в примере 1.1).
#------------------------------------------------------------------

import numpy as np
from mpi4py import MPI


#------------------------------------------------------------------
def auxiliary_arrays_determination(M, P):
    """
    Расчёт списков числа элементов `rcounts` и соответствующих
    смещений "displs", определяющих распределение больших матриц
    и векторов по процессам MPI коммуникатора.

    :param M: Общее число элементов вдоль нужной оси матрицы.
    :param P: Общее число процессов, работающих над параллелизацией
              вычислений. Предполагаем при этом, что поток 0 только
              "дирижирует" работу, а "рабочие" потоки от 1 до P-1
              её выполняют.
    :return: Рассчитанные списки числа элементов `rcounts` и
             соответствующих смещений "displs", определяющие
             передачу данных каждому процессу.
    """
    # Найдём целые числа K и L из описания алгоритма выше:
    K, L = divmod(np.int32(M), P - 1)

    # Введём два новых списка для описания того, как именно
    # матрицы и векторы будут распределяться по всем процессам.
    # Здесь `rcounts` будет содержать число элементов, хранимое
    # каждым процессом (это 0 для процесса 0, K+1 для первых L
    # "рабочих" процессов, и K для оставшихся процессов).
    # Другой список `displs` будет содержать индекс смещений
    # - то есть, номер первой строки, начиная с которой будут
    # храниться `rcounts[m]` строк на процессе `m`.
    # При этом мы предполагаем, что все элементы, которые
    # хранятся на каждом процессе, идут подряд.
    rcounts = np.empty(P, dtype=np.int32)
    displs = np.empty(P, dtype=np.int32)

    # Процесс 0 не рабочий, и он будет содержать ноль элементов:
    rcounts[0] = displs[0] = 0

    # Цикл по всем "рабочим" процессам:
    for m in range(1, P):
        if m <= L:
            rcounts[m] = K + 1
        else:
            rcounts[m] = K
        # Индекс смещений сдвигается каждый раз на число
        # строк, хранимых в процессе:
        displs[m] = displs[m - 1] + rcounts[m - 1]
    return rcounts, displs


#------------------------------------------------------------------
def conjugate_gradient_method(A_part, b_part, x_part,
                              N, N_part, rcounts_N, displs_N):
    """
    Реализация параллельного итерационного алгоритма решения систем линейных
    уравнений методом сопряжённых градиентов.

    :param A_part:
    :param b_part:
    :param x_part:
    :param N:
    :param N_part:
    :param rcounts_N:
    :param displs_N:

    :return: x_part
    """

    x = np.empty(N, dtype=np.float64)
    p = np.empty(N, dtype=np.float64)

    r_part = np.empty(N_part, dtype=np.float64)
    p_part = np.empty(N_part, dtype=np.float64)
    q_part = np.empty(N_part, dtype=np.float64)

    ScalP = np.array(0, dtype=np.float64)
    ScalP_temp = np.empty(1, dtype=np.float64)

    s = 1

    p_part = 0.

    while s <= N:

        if s == 1:
            comm.Allgatherv([x_part, N_part, MPI.DOUBLE],
                            [x, rcounts_N, displs_N, MPI.DOUBLE])
            r_temp = np.dot(A_part.T, np.dot(A_part, x) - b_part)
            comm.Reduce_scatter([r_temp, N, MPI.DOUBLE],
                                [r_part, N_part, MPI.DOUBLE],
                                recvcounts=rcounts_N, op=MPI.SUM)
        else:
            ScalP_temp[0] = np.dot(p_part, q_part)
            comm.Allreduce([ScalP_temp, 1, MPI.DOUBLE],
                           [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
            r_part = r_part - q_part/ScalP

        ScalP_temp[0] = np.dot(r_part, r_part)
        comm.Allreduce([ScalP_temp, 1, MPI.DOUBLE],
                       [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
        p_part = p_part + r_part/ScalP

        comm.Allgatherv([p_part, N_part, MPI.DOUBLE],
                        [p, rcounts_N, displs_N, MPI.DOUBLE])
        q_temp = np.dot(A_part.T, np.dot(A_part, p))
        comm.Reduce_scatter([q_temp, N, MPI.DOUBLE],
                            [q_part, N_part, MPI.DOUBLE],
                            recvcounts=rcounts_N, op=MPI.SUM)

        ScalP_temp[0] = np.dot(p_part, q_part)
        comm.Allreduce([ScalP_temp, 1, MPI.DOUBLE],
                       [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
        x_part = x_part - p_part/ScalP

        s = s + 1

    return x_part


#------------------------------------------------------------------
# Начинаем выполнение программы - первым делом, настроим MPI:
#------------------------------------------------------------------

# Работаем с коммуникатором по всем доступным процессам:
comm = MPI.COMM_WORLD

# Число P доступных процессов в этом коммуникаторе:
P = comm.Get_size()

# Номер текущего процесса (от 0 до P-1):
rank = comm.Get_rank()

#------------------------------------------------------------------
# Шаг 1 (точно такой же, как в примерах 1.1 и 2.2):
# Зачитаем из файла размер (M, N) матрицы A.
# Сделаем это только на процессе 0 - считаем,
# что входной файл доступен только на нём:

if rank == 0:
    with open('Example-03-1_in.dat', 'r') as f1:
        M = np.array(np.int32(f1.readline()))
        N = np.array(np.int32(f1.readline()))
else:
    # На "рабочих" процессах значение `M` не
    # используется - ставим пустую "заглушку":
    M = None
    # и подготавливаем на них "хранилище" для `N`:
    N = np.array(0, dtype=np.int32)

# Раздадим значение `N`, зачитанное процессом 0,
# всем остальным процессам (включая и сам процесс 0):
comm.Bcast(N, root=0)
# Альтернативно, более общая форма записи:
# comm.Bcast([N, 1, MPI.INT], root=0)

# NOTE: Значение `M` не используется на "рабочих" процессах,
#       так что мы его не будем раздавать - лишняя трата времени.

#------------------------------------------------------------------
# Шаг 2:
# Разберёмся, сколько строк (и какие именно!) из матрицы `A` и векторов
# `b` и `x` мы будем хранить в форме частичной матрицы `A_part` и частичных
# векторов `b_part` и `x_part` на каждом из "рабочих" процессов.
# Сделаем такой анализ только на процессе 0:

if rank == 0:
    # Как обычно, введём списки числа строк/элементов и их
    # смещения для описания того, как именно матрица и векторы
    # распределяются по всем процессам. Но на этот раз нам
    # понадобятся такие списки для обоих осей матрицы.
    # Всю логику расчёта этих списков мы перенесём в отдельную
    # функцию `auxiliary_arrays_determination()`, определённую
    # выше в этом файле:
    rcounts_M, displs_M = auxiliary_arrays_determination(M, P)
    rcounts_N, displs_N = auxiliary_arrays_determination(N, P)
else:
    # На "рабочих" процессах список смещений `displs_M`
    # не используется и значит не нужен - а из списка
    # `rcounts_M` каждый процесс использует только одно
    # число - ниже мы его назовём `M_part`, так что
    # сам список `rcounts_M` тоже не понадобится:
    rcounts_M = displs_M = None
    # Подготовим "хранилища" для других двух списков:
    rcounts_N = np.empty(P, dtype=np.int32)
    displs_N = np.empty(P, dtype=np.int32)

# Подготовим "хранилище" для `M_part` на всех процессах:
M_part = np.array(0, dtype=np.int32)
N_part = np.array(0, dtype=np.int32)
# NOTE: Отдельного значения типа N_part мы готовить не будем,
# поскольку каждый процесс получит полную копию `rcounts_N`.

# Разбросаем рассчитанные выше `rcounts_M` по всем процессам
# в виде одного значения `M_part = rcounts_M[m]` на каждом
# процессе `m`:
comm.Scatter([rcounts_M, 1, MPI.INT],
             [M_part, 1, MPI.INT], root=0)

# Передадим полные версии списков `rcounts_N` и `displs_N`
# всем процессам:
comm.Bcast([rcounts_N, P, MPI.INT], root=0)
comm.Bcast([displs_N, P, MPI.INT], root=0)

#------------------------------------------------------------------
# Шаг 3:
# Зачитаем из файла матрицу `A`. Как и в примере 2.2, мы будем
# экономить память - будем зачитывать матрицу по кусочкам,
# сразу отдавая каждый кусочек нужному процессу.

# Подготовим "хранилище" для кусочков `A_part` на всех процессах -
# каждое со своим числом строк `M_part`:
A_part = np.empty((M_part, N), dtype=np.float64)

if rank == 0:
    # Зачитаем на процессе 0 файл с матрицей A не сразу весь,
    # а по кусочкам - сразу отдавая каждый кусочек своему
    # "рабочему" процессу (на процессе 0 при этом данных
    # матрицы A совсем не останется - экономим память!):
    with open('Example-03-1_AData.dat', 'r') as f2:
        for m in range(1, P):
            # Кусочек матрицы A, который мы отдадим процессу m:
            A_part_m = np.empty((rcounts_M[m], N), dtype=np.float64)
            # Зачитали данные:
            for j in range(rcounts_M[m]):
                for i in range(N):
                    A_part_m[j,i] = np.float64(f2.readline())
            # И сразу же отдали их процессу, причём без блокировки:
            comm.Send([A_part_m, rcounts_M[m]*N, MPI.DOUBLE], dest=m, tag=0)
            # Такая же, но блокирующая пересылка:
            #comm.Send([A_part_m, rcounts_M[m]*N, MPI.DOUBLE], dest=m, tag=0)
            # Поможем сборщику мусора поскорее избавиться от ненужных данных:
            del A_part_m
else:
    # Каждый "рабочий" процесс получает свою часть строк и записывает
    # их в свой массив `A_part` - теперь можно и нужно с блокировкой:
    comm.Recv([A_part, M_part*N, MPI.DOUBLE], source=0, tag=0, status=None)

#------------------------------------------------------------------
# Шаг 4:
# Зачитаем из файла вектор `b`. Зачитаем его на процессе 0
# (считаем, что входной файл доступен только на нём) - и потом
# раздадим его по кусочки `b_part` всем "рабочим" процессам.

# Зачитаем файл `b` на процессе 0:
if rank == 0:
    b = np.empty(M, dtype=np.float64)
    with open('Example-03-1_bData.dat', 'r') as f3:
        for j in range(M):
            b[j] = np.float64(f3.readline())
else:
    # На "рабочих" процессах вектор `b` не используется
    # и значит не нужен:
    b = None

# Подготовим "хранилище" для `b_part` на всех процессах:
b_part = np.empty(M_part, dtype=np.float64)

# И разбросаем зачитанный выше вектор `b` по всем процессам
# в виде кусочков `b_part` (с размером `M_part`) этого вектора:
comm.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE], 
              [b_part, M_part, MPI.DOUBLE], root=0)

#------------------------------------------------------------------
# Шаг 5:
# Подготовим "хранилища" для вектора `x`.
# Полная версия вектора `x` будет храниться только на процессе 0:
if rank == 0:
    x = np.zeros(N, dtype=np.float64)
else:
    x = None

# На рабочих же процессах будут храниться только кусочки вектора `x`
# в виде векторов `x_part` (который будет создан и на процессе 0,
# но с нулевым размером).

# Подготовим "хранилище" для `x_part` на всех процессах:
x_part = np.empty(rcounts_N[rank], dtype=np.float64)

# И разбросаем вектор `x` по всем процессам в виде кусочков
# `x_part` (с размером `M_part`) этого вектора:
comm.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE],
              [x_part, rcounts_N[rank], MPI.DOUBLE], root=0)

#------------------------------------------------------------------
# Шаг 6:
# Собственно и сама нужная нам работа - решение системы линейных
# уравнений итерационным методом градиентного спуска:

x_part = conjugate_gradient_method(A_part, b_part, x_part,
                                   N, rcounts_N[rank], rcounts_N, displs_N)

print(f"Vector `x_part` on process {rank} consists of {len(x_part)} elements:")
print(f"  x_part = {x_part}")

# Соберём вектор `x` на процессе 0 (root) из кусочков `x_part`, присланными
# всеми процессами (включая и пустой кусочек от самого процесса 0):
comm.Gatherv([x_part, rcounts_N[rank], MPI.DOUBLE],
             [x, rcounts_N, displs_N, MPI.DOUBLE], root=0)

#------------------------------------------------------------------
# Шаг 7:
# Окончательно, нарисуем посчитанный вектор `x`, используя
# библиотеку matplotlib. Делаем это только на процессе 0.

if rank == 0:
    # Для контроля, напечатаем найденное решение на консоли:
    print(f"x = {x}")

    # Подготовим рисунок и данные для него:
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    fig = plt.figure()
    ax = plt.axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i'); ax.set_ylabel('x[i]')
    # индексы элементов вектора `x`:
    ii = np.arange(np.int32(N))

    # Нарисуем полное решение:
    #ax.plot(ii, x, '-y', lw=3)

    # А лучше, нарисуем кусочки, которые были посчитаны
    # на каждом отдельном процессе (поиграйтесь с числом
    # процессов!):
    for m in range(1, P):
        r = rcounts_N[m]
        d = displs_N[m]
        ax.plot(ii[d:d+r], x[d:d+r], '-', lw=3,
                label=f"Process {m}")
    plt.legend()
    plt.show()

#------------------------------------------------------------------
