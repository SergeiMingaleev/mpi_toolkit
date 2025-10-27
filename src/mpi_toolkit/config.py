import numpy as np

class _Config:
    def __init__(self):
        self._locked = False
        self._numpy_lib = np

    def lock(self):
        """
        Lock critical default settings (such as the used numpy_lib) so that
        they cannot be modified anymore after calling this method.
        """
        self._locked = True

    @property
    def numpy_lib(self):
        return self._numpy_lib
    
    @numpy_lib.setter
    def numpy_lib(self, val):
        if self._locked:
            raise RuntimeError(
                'Variable "mpi_toolkit.config.numpy_lib" is now protected. '
                'It may be changed only immediately after importing '
                '"mpi_toolkit" library')
        # TODO: check that it is indeed a valid numpy-like module.
        self._numpy_lib = val

config = _Config()
