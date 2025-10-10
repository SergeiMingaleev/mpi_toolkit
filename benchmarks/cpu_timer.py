import numpy as np
from mpi4py import MPI


#------------------------------------------------------------------
class CPU_Timer:
    """
    CPU_Timer class for benchmarking performance of
    the selected parts of the code. The time is
    shown in seconds assuming a 1 GHz CPU speed and
    should be more or less the same when executing
    on different ix86 computers. This timer is not,
    however, really reliable for accurate benchmarking
    because it shows a full execution time (not the
    CPU time used for this particular problem).

    NOTE: the CPU_Timer can be reliably used for only
    benchmarking of rather slow subroutines (whose
    execution time exceeds few mksec).

    USAGE:
        timer = CPU_Timer()
        timer.start()
        .............
        timer.stop()

        timer.show("code_name")
        time = timer.time()

        timer.restart()
    """
    def __init__(self):
        self._time_sum = 0.0
        self._time_start = MPI.Wtime()
        self._timer_switched_on = False

    def start(self):
        if not self._timer_switched_on:
            self._time_start = MPI.Wtime()
            self._timer_switched_on = True

    def restart(self):
        self._time_sum = 0.0
        self._time_start = MPI.Wtime()
        self._timer_switched_on = True

    def stop(self):
        if self._timer_switched_on:
            t_stop = MPI.Wtime()
            self._time_sum += t_stop - self._time_start
            self._timer_switched_on = False

    def time(self):
        time_add = 0.0
        if self._timer_switched_on:
            time_add = MPI.Wtime() - self._time_start
        return self._time_sum + time_add

    def show(self, info):
        print(f"Elapsed time for '{info}': {self.time()} sec.")


#------------------------------------------------------------------
class CPU_TimerMap:
    """
    CPU_TimerMap class for flexible benchmarking
    of the performance of different frequently used
    subroutines.

    This class is based on the CPU_Timer class - look at its
    code for more detailed comments on the performance.

    USAGE:
        GlobalTimer = CPU_TimerMap() # this class should usually
                                     # be used as a global timer.
        GlobalTimer.start("id")
        .................
        GlobalTimer.stop("id")

        GlobalTimer.show("id")
        time_id = GlobalTimer.time("id")

        GlobalTimer.restart("id")

        GlobalTimer.show()  # show a summary for all timers.
    """
    def __init__(self, comm=None):
        if comm is None:
            comm = MPI.COMM_WORLD
        self._comm = comm
        self._P = comm.Get_size()
        self._rank = comm.Get_rank()

        self._timer = dict()  # key: str, value: CPU_Timer
        self._number_of_calls = dict()  # key: str, value: int

    def add(self, id):
        self._timer[id] = CPU_Timer()
        self._number_of_calls[id] = 0

    def start(self, id):
        if not id in self._timer:
            self._timer[id] = CPU_Timer()
            self._number_of_calls[id] = 0
        self._timer[id].start()
        self._number_of_calls[id] += 1

    def start_fast(self, id):
        self._timer[id].start()
        self._number_of_calls[id] += 1

    def restart(self, id):
        self._timer[id].restart()
        self._number_of_calls[id] = 1

    def stop(self, id):
        self._timer[id].stop()

    def time(self, id):
        return self._timer[id].time()

    def show(self, id = None):
        if id is not None:
            self._timer[id].show(id)

        sorted_by_time = dict(sorted(self._timer.items(),
                                     key=lambda item: item[1].time(),
                                     reverse=True))

        max_time_ms = 1e3 * np.max([item.time() for item in self._timer.values()])
        id_max_len = np.max(list(map(len, self._timer.keys())))
        pp = f"[{self._rank}/{self._P}]"

        print(f"{pp} ============================= CPU_Timer Summary: ===============================")
        for id in sorted_by_time:
            t = 1e3 * sorted_by_time[id].time() # ms
            N = self._number_of_calls[id]

            print(f"{pp} {t:10,.4f} ms "
                  f"({100 * t / max_time_ms:7,.3f} %)"
                  f" -- {id.ljust(id_max_len)} "
                  f"-- {N:6} calls == "
                  f"{t / N:11,.5f} ms/call")
        print(f"{pp} ================================================================================")

#------------------------------------------------------------------
