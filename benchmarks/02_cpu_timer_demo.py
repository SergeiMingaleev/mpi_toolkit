#------------------------------------------------------------------
# Демонстрация использования самодельного класса `BenchmarkTimer`
# и проверка скорости его работы.
#
# Описание работы класса смотри в файле
# mpi_toolkit/benchmarks/linalg_solvers/cpu_timer.py
# с реализацией этого класса.
#------------------------------------------------------------------

from mpi4py import MPI
from linalg_solvers import BenchmarkTimer

N = 1_000_000

#------------------------------------------------------------------
mpi_timer = BenchmarkTimer()

mpi_timer.start("empty cycles")
for i in range(N):
    pass
mpi_timer.stop("empty cycles")

#------------------------------------------------------------------

x = 2.0
y = 1.17

mpi_timer.start("arithmetic cycles")
for i in range(N):
    x = x * y
mpi_timer.stop("arithmetic cycles")

#------------------------------------------------------------------

mpi_timer.start("MPI.Wtime cycles")
for i in range(N):
    t = MPI.Wtime()
mpi_timer.stop("MPI.Wtime cycles")

#------------------------------------------------------------------
cpu_timer_test = BenchmarkTimer()

mpi_timer.start("BenchmarkTimer.start")
for i in range(N):
    cpu_timer_test.start("BenchmarkTimer.start")
mpi_timer.stop("BenchmarkTimer.start")

mpi_timer.start("BenchmarkTimer.stop")
for i in range(N):
    cpu_timer_test.start("BenchmarkTimer.stop")
mpi_timer.stop("BenchmarkTimer.stop")

#------------------------------------------------------------------
cpu_timer_test = BenchmarkTimer()

mpi_timer.start("BenchmarkTimer.start/stop")
for i in range(N):
    cpu_timer_test.start("BenchmarkTimer.start/stop")
    cpu_timer_test.stop("BenchmarkTimer.start/stop")
mpi_timer.stop("BenchmarkTimer.start/stop")

#------------------------------------------------------------------

mpi_timer.print_status()
cpu_timer_test.print_status()

#------------------------------------------------------------------
# ВЫВОД:
# На Windows использование пары команд `BenchmarkTimer.start(msg)`
# и `BenchmarkTimer.stop(msg)` занимает в ~2 раза больше времени,
# чем два вызова `mpi4py.MPI.Wtime()`, и соответствует ~4
# арифметическим операциям.
#
# Плата за удобство организации таймеров минимальная -
# можно пользоваться! :-)
#------------------------------------------------------------------
