import time
import numpy as np

try:
    # Если `mpi4py` установлен, работаем с его счётчиком времени,
    # он чуть быстрее (примерно на 30%) на платформе Windows:
    from mpi4py import MPI
    get_time_now = MPI.Wtime
except ModuleNotFoundError:
    # Иначе работаем с `time.perf_counter`, который является самым
    # быстрым и точным из стандартных счётчиков времени Python:
    MPI = None
    get_time_now = time.perf_counter

#------------------------------------------------------------------
class StopwatchTimer:
    """
    StopwatchTimer class for benchmarking performance of
    the selected parts of the code. The time is
    shown in seconds assuming a 1 GHz CPU speed and
    should be more or less the same when executing
    on different ix86 computers. This timer is not,
    however, really reliable for accurate benchmarking
    because it shows a full execution time (not the
    CPU time used for this particular problem).

    NOTE: the StopwatchTimer can be reliably used for only
    benchmarking of rather slow subroutines (whose
    execution time exceeds few mksec).

    USAGE:
        timer = StopwatchTimer()
        timer.start()
        .............
        timer.stop()

        dt = timer.duration()

        timer.print_status()

        timer.reset()
    """
    def __init__(self):
        self._timer_switched_on = False
        self._time_start = None
        self._duration = 0.0
        self._ncalls = 0

    def reset(self):
        """
        Обнулить и выключить таймер.
        """
        self.__init__()

    def start(self):
        """
        Включить таймер (если он был выключен)
        и увеличить счётчик вызовов таймера на единицу.
        """
        self._ncalls += 1
        if not self._timer_switched_on:
            self._time_start = get_time_now()
            self._timer_switched_on = True

    def stop(self):
        """
        Выключить таймер (если он был включён).
        """
        if self._timer_switched_on:
            t_stop = get_time_now()
            self._duration += t_stop - self._time_start
            self._timer_switched_on = False

    @property
    def ncalls(self):
        """
        Число включений таймера (то есть, число вызовов метода `start()`).

        :return: (int) Число вызовов.
        """
        return self._ncalls

    def duration(self):
        """
        Полное время работы таймера.

        :return: (float) Время в секундах.
        """
        time_add = 0.0
        if self._timer_switched_on:
            time_add = get_time_now() - self._time_start
        return self._duration + time_add

    def print_status(self):
        """
        Напечатать информацию о текущем состоянии таймера.
        """
        t = 1e3 * self.duration()
        N = self.ncalls
        print(f"Elapsed time: {t:10,.4f} ms "
              f"({N} calls == {t / N:11,.5f} ms/call)")


#------------------------------------------------------------------
class BenchmarkTimer:
    """
    BenchmarkTimer class for flexible benchmarking
    of the performance of different frequently used
    subroutines.

    This class is based on the StopwatchTimer class - look at its
    code for more detailed comments on the performance.

    USAGE:
        timer = BenchmarkTimer() # this class should usually
                                 # be used as a global timer.
        timer.start("tag")
        .................
        timer.stop("tag")

        timer.is_active = False / True
        time_tag = timer.duration("tag")
        timer.reset("tag")

        timer.print_status()
    """
    def __init__(self, comm=None):
        if MPI is None:
            self._comm = self._P = self._rank = None
        else:
            if comm is None:
                comm = MPI.COMM_WORLD
            self._comm = comm
            self._P = comm.Get_size()
            self._rank = comm.Get_rank()

        self._timers = dict()  # key: str, value: StopwatchTimer
        self._disabled = False

    @property
    def is_active(self):
        """
        Флаг, определяющий, является ли таймер активным.
        В неактивном состоянии вызовы всех методов класса
        будут молча игнорироваться, с минимальным влиянием
        на скорость выполнения программы.

        :getter: Возвращает значение, является ли таймер
                 активным.
        :setter: Устанавливает состояние активности таймера.

        :type: bool
        """
        return not self._disabled

    @is_active.setter
    def is_active(self, value):
        self._disabled = not bool(value)

    def reset(self, tag=None):
        """
        Обнуляет таймер для указанного `tag`.
        Вызов метода без параметров удалит все
        созданные таймеры и их накопленные данные.
        """
        if self._disabled:
            return
        if tag is None:
            self.__init__(comm=self._comm)
        else:
            self._timers[tag].reset()

    def start(self, tag):
        """
        Запустить таймер с указанным ярлыком.
        При запуске будет произведена проверка, существует ли уже
        таймер с указанным ярлыком - и если нет, то он будет создан.

        :param tag: (str) Ярлык запускаемого таймера.
        """
        if self._disabled:
            return
        if tag not in self._timers:
            self._timers[tag] = StopwatchTimer()
        self._timers[tag].start()

    def stop(self, tag):
        """
        Остановить (поставить на паузу) таймер с указанным ярлыком.
        При запуске будет произведена проверка, существует ли уже
        таймер с указанным ярлыком - и если нет, То он будет создан.
        Альтернативный более быстрый запуск таймера (но без проверки)
        доступен через метод `start_fast(tag)`.
        :param tag: (str) Ярлык запускаемого таймера.
        """
        if self._disabled:
            return
        self._timers[tag].stop()

    def duration(self, tag):
        """

        :param tag: (str) Ярлык добавляемого таймера.
        :return:
        """
        if self._disabled:
            return 0.0
        return self._timers[tag].duration()

    def print_status(self):
        """
        Напечатать информацию о текущем состоянии таймера.
        """
        if self._disabled:
            return
        sorted_by_time = dict(sorted(self._timers.items(),
                                     key=lambda item: item[1].duration(),
                                     reverse=True))

        max_time_ms = 1e3 * np.max([item.duration() for item in self._timers.values()])
        tag_max_len = np.max(list(map(len, self._timers.keys())))
        if self._P is None:
            pp = ""
        else:
            pp = f"[{self._rank}/{self._P}]"

        print(f"{pp} ============================= BenchmarkTimer Summary: ===============================")
        for tag in sorted_by_time:
            t = 1e3 * sorted_by_time[tag].duration() # ms
            N = sorted_by_time[tag]._ncalls

            print(f"{pp} {t:10,.4f} ms "
                  f"({100 * t / max_time_ms:7,.3f} %): "
                  f"{tag.ljust(tag_max_len)} "
                  f"-- {N:6} calls == "
                  f"{t / N:11,.5f} ms/call")
        print(f"{pp} ================================================================================")

#------------------------------------------------------------------
