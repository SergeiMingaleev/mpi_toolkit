from mpi4py import MPI

from ..config import config

# Работаем с коммуникатором по всем доступным процессам:
comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()


def arange(*args, rank=0, numpy_lib=None, **kwargs):
    if numpy_lib is None:
        # Нельзя больше изменять `config.numpy_lib`:
        config.lock()
        numpy_lib = config.numpy_lib

    if rank != comm_rank:
        return numpy_lib.nan
    return numpy_lib.arange(*args, **kwargs)


def sin(*args, rank=0, numpy_lib=None, **kwargs):
    if numpy_lib is None:
        # Нельзя больше изменять `config.numpy_lib`:
        config.lock()
        numpy_lib = config.numpy_lib

    if rank != comm_rank:
        return numpy_lib.nan
    return numpy_lib.sin(*args, **kwargs)


def zeros(*args, rank=0, numpy_lib=None, **kwargs):
    if numpy_lib is None:
        # Нельзя больше изменять `config.numpy_lib`:
        config.lock()
        numpy_lib = config.numpy_lib

    if rank != comm_rank:
        return numpy_lib.nan
    return numpy_lib.zeros(*args, **kwargs)


def dot(*args, rank=0, numpy_lib=None, **kwargs):
    if numpy_lib is None:
        # Нельзя больше изменять `config.numpy_lib`:
        config.lock()
        numpy_lib = config.numpy_lib

    if rank != comm_rank:
        return numpy_lib.nan
    return numpy_lib.dot(*args, **kwargs)


def error_max_abs(x, y, rank=0, numpy_lib=None):
    if numpy_lib is None:
        # Нельзя больше изменять `config.numpy_lib`:
        config.lock()
        numpy_lib = config.numpy_lib

    if rank != comm_rank:
        return numpy_lib.nan
    return numpy_lib.max(numpy_lib.abs(x - y))


