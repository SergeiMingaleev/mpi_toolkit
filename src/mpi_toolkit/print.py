from builtins import print as print_builtin
from mpi4py import MPI


# Работаем с коммуникатором по всем доступным процессам:
comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()


def print(*values: object,
          sep: str | None = " ",
          end: str | None = "\n",
          rank: int = 0) -> None:
    """
    Prints the values to sys.stdout at the MPI process
    with the given `rank`.

    :param values: values to be prineted.
    :param sep: string inserted between values, default a space.
    :param end: string appended after the last value, default a newline.
    :param rank: rank of the MPI process, at which the values should be printed,
                 default zero process (which always exists, and is recommended
                 to be used as the main orchestrating process).
    """
    if rank == comm_rank:
        print_builtin(*values, sep=sep, end=end)

def print_all(*values: object,
          sep: str | None = " ",
          end: str | None = "\n") -> None:
    """
    Prints the values to sys.stdout at all the MPI processes.

    :param values: values to be prineted.
    :param sep: string inserted between values, default a space.
    :param end: string appended after the last value, default a newline.
    """
    print_builtin(*values, sep=sep, end=end)
