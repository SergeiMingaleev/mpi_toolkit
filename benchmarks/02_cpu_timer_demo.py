#------------------------------------------------------------------
# Демонстрация использования самодельного класса `CPU_TimerMap`
# и проверка скорости его работы.
#
# Описание работы класса смотри в файле "cpu_timer.py"
# с реализацией этого класса.
#------------------------------------------------------------------

from mpi4py import MPI
from cpu_timer import CPU_TimerMap

N = 1_000_000

#------------------------------------------------------------------
mpi_timer = CPU_TimerMap()

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
cpu_timer_test = CPU_TimerMap()

mpi_timer.start("CPU_Timer start")
for i in range(N):
    cpu_timer_test.start("CPU_Timer start")
mpi_timer.stop("CPU_Timer start")

cpu_timer_test.add("CPU_Timer start_fast")

mpi_timer.start("CPU_Timer start_fast")
for i in range(N):
    cpu_timer_test.start_fast("CPU_Timer start_fast")
mpi_timer.stop("CPU_Timer start_fast")

mpi_timer.start("CPU_Timer stop")
for i in range(N):
    cpu_timer_test.start("CPU_Timer stop")
mpi_timer.stop("CPU_Timer stop")

#------------------------------------------------------------------
cpu_timer_test = CPU_TimerMap()

mpi_timer.start("CPU_Timer start/stop")
for i in range(N):
    cpu_timer_test.start("CPU_Timer start/stop")
    cpu_timer_test.stop("CPU_Timer start/stop")
mpi_timer.stop("CPU_Timer start/stop")

#------------------------------------------------------------------
cpu_timer_test.add("CPU_Timer start_fast/stop")

mpi_timer.start("CPU_Timer start_fast/stop")
for i in range(N):
    cpu_timer_test.start_fast("CPU_Timer start_fast/stop")
    cpu_timer_test.stop("CPU_Timer start_fast/stop")
mpi_timer.stop("CPU_Timer start_fast/stop")

#------------------------------------------------------------------

mpi_timer.show()
cpu_timer_test.show()

#------------------------------------------------------------------
# ВЫВОД:
# На Windows использование пары команд `CPU_TimerMap.start(msg)`
# и `CPU_TimerMap.stop(msg)` занимает в ~2 раза больше времени,
# чем два вызова `mpi4py.MPI.Wtime()`, и соответствует ~4
# арифметическим операциям.
#
# Использование же пары команд `CPU_TimerMap.start_fast(msg)`
# # и `CPU_TimerMap.stop(msg)` соответствует ~3 арифметическим
# операциям.
#
# Плата за удобство организации таймеров минимальная -
# можно пользоваться! :-)
#------------------------------------------------------------------
