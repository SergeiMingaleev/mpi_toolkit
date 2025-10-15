#------------------------------------------------------------------
# Проверка двух высокоточных счётчиков времени, доступных в Python.
#
# Первый счётчик встроен в стандартную библиотеку Python:
#   import time
#   t = time.perf_counter()
# Тот же счётчик, но возвращает значение типа int (вместо float)
# в наносекундах (вместо секунд):
#   t_ns = time.perf_counter_ns()
#
# Второй счётчик - это обёртка над вызовом `MPI_Wtime()`:
#   from mpi4py import MPI
#   t = MPI.Wtime()
#
# На системе Windows, оба они под капотом основаны на одной и той же
# функции, доступной в C/C++ как `QueryPerformanceCounter()`.
# Детали использования и реализации этой функции обсуждаются на странице:
# https://learn.microsoft.com/ru-ru/windows/win32/api/profileapi/nf-profileapi-queryperformancecounter
# https://learn.microsoft.com/ru-ru/windows/win32/sysinfo/acquiring-high-resolution-time-stamps
#
# Однако из-за особенностей реализации, потенциально возможны разные
# накладные расходы на использование каждого из счётчиков - проверим это!
#------------------------------------------------------------------


import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
P = comm.Get_size()
rank = comm.Get_rank()

N = 1_000_000

if rank == 0:
    print(f"Number of MPI processes: {P}")
    print(f"N = {N}")
    print()

    # Доступная информация про `time.perf_counter()`:
    clock_info = time.get_clock_info('perf_counter')
    print("Info for the function `time.perf_counter()`:")
    print(f"   resolution: {clock_info.resolution} seconds")
    print(f"   monotonic:  {clock_info.monotonic}")
    print(f"   adjustable: {clock_info.adjustable}")
    print(f"   implementation: {clock_info.implementation}")
    print()

    # Доступная информация про `mpi4py.MPI.Wtime()`:

    print("Info for the function `mpi4py.MPI.Wtime()`:")
    print(f"   resolution: {MPI.Wtick()} seconds")
    print()

#------------------------------------------------------------------

t0_pc_ns = time.perf_counter_ns()
for i in range(N):
    pass
t1_pc_ns = time.perf_counter_ns()

t0_pc = time.perf_counter()
for i in range(N):
    pass
t1_pc = time.perf_counter()

t0_mpi = MPI.Wtime()
for i in range(N):
    pass
t1_mpi = MPI.Wtime()

dt_pc_ns = t1_pc_ns - t0_pc_ns
dt_pc = t1_pc - t0_pc
dt_mpi = t1_mpi - t0_mpi

print("Timing for empty cycles:")
print(f"   time.perf_counter_ns(): {dt_pc_ns} ns ({1e-9*dt_pc_ns} sec) on process {rank}")
print(f"   time.perf_counter(): {dt_pc} seconds on process {rank}")
print(f"   mpi4py.MPI.Wtime():  {dt_mpi} seconds on process {rank}")
print()

#------------------------------------------------------------------

x = 2.0
y = 1.17

t0_pc_ns = time.perf_counter_ns()
for i in range(N):
    x = x * y
t1_pc_ns = time.perf_counter_ns()

t0_pc = time.perf_counter()
for i in range(N):
    x = x * y
t1_pc = time.perf_counter()

t0_mpi = MPI.Wtime()
for i in range(N):
    x = x * y
t1_mpi = MPI.Wtime()

dt_pc_ns = t1_pc_ns - t0_pc_ns
dt_pc = t1_pc - t0_pc
dt_mpi = t1_mpi - t0_mpi

print("Timing for arithmetic cycles:")
print(f"   time.perf_counter_ns(): {dt_pc_ns} ns ({1e-9*dt_pc_ns} sec) on process {rank}")
print(f"   time.perf_counter(): {dt_pc} seconds on process {rank}")
print(f"   mpi4py.MPI.Wtime():  {dt_mpi} seconds on process {rank}")
print()

#------------------------------------------------------------------

t0_pc_ns = time.perf_counter_ns()
for i in range(N):
    t = time.perf_counter_ns()
t1_pc_ns = time.perf_counter_ns()

t0_pc = time.perf_counter()
for i in range(N):
    t = time.perf_counter()
t1_pc = time.perf_counter()

t0_mpi = MPI.Wtime()
for i in range(N):
    t = MPI.Wtime()
t1_mpi = MPI.Wtime()

dt_pc_ns = t1_pc_ns - t0_pc_ns
dt_pc = t1_pc - t0_pc
dt_mpi = t1_mpi - t0_mpi

print("Timing for time cycles:")
print(f"   time.perf_counter_ns(): {dt_pc_ns} ns ({1e-9*dt_pc_ns} sec) on process {rank}")
print(f"   time.perf_counter(): {dt_pc} seconds on process {rank}")
print(f"   mpi4py.MPI.Wtime():  {dt_mpi} seconds on process {rank}")
print()

print("Estimated time for one time counter call:")
print(f"   time.perf_counter_ns(): {dt_pc_ns/N} ns ({1e-9*dt_pc_ns/N} sec) on process {rank}")
print(f"   time.perf_counter(): {dt_pc/N} seconds on process {rank}")
print(f"   mpi4py.MPI.Wtime():  {dt_mpi/N} seconds on process {rank}")
print()

#------------------------------------------------------------------
# ВЫВОД: На Windows вызов `mpi4py.MPI.Wtime()` занимает меньше
# времени, чем вызов `time.perf_counter()` (примерно на 30%).
# Вызов же `time.perf_counter_ns()` оказался самым долгим.
# Используем дальше `mpi4py.MPI.Wtime()`!
#------------------------------------------------------------------
