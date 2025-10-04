#------------------------------------------------------------------
# Пример 2.2: Параллельное решение системы линейных уравнений
#             A*x = b
#
# Задача: найти вектор x, являющийся решением системы линейных
#         уравнений A*x = b
#------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

# Работаем с коммуникатором по всем доступным процессам:
comm = MPI.COMM_WORLD

# Число P доступных процессов в этом коммуникаторе:
P = comm.Get_size()

# Номер текущего процесса (от 0 до P-1):
rank = comm.Get_rank()


#------------------------------------------------------------------
def conjugate_gradient_method(A_part, b_part, x_part,
                              N, N_part, rcounts_N, displs_N):

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
if rank == 0:
    with open('Example-03-1_in.dat', 'r') as f1:
        N = np.array(np.int32(f1.readline()))
        M = np.array(np.int32(f1.readline()))
else:
    N = np.array(0, dtype=np.int32)

comm.Bcast([N, 1, MPI.INT], root=0)


#------------------------------------------------------------------
def auxiliary_arrays_determination(M, P):
    K, L = divmod(M, P-1)
    rcounts = np.empty(P, dtype=np.int32)
    displs = np.empty(P, dtype=np.int32)
    rcounts[0] = displs[0] = 0
    for m in range(1, P):
        if m <= L:
            rcounts[m] = K + 1
        else:
            rcounts[m] = K
        displs[m] = displs[m-1] + rcounts[m-1]
    return rcounts, displs

#------------------------------------------------------------------
if rank == 0:
    rcounts_M, displs_M = auxiliary_arrays_determination(M, P)
    rcounts_N, displs_N = auxiliary_arrays_determination(N, P)
else:
    rcounts_M = displs_M = None
    rcounts_N = np.empty(P, dtype=np.int32)
    displs_N = np.empty(P, dtype=np.int32)

M_part = np.array(0, dtype=np.int32)
N_part = np.array(0, dtype=np.int32)

comm.Scatter([rcounts_M, 1, MPI.INT],
             [M_part, 1, MPI.INT], root=0)

comm.Bcast([rcounts_N, P, MPI.INT], root=0)
comm.Bcast([displs_N, P, MPI.INT], root=0)

if rank == 0:
    with open('Example-03-1_AData.dat', 'r') as f2:
        for k in range(1, P):
            A_part = np.empty((rcounts_M[k], N), dtype=np.float64)
            for j in range(rcounts_M[k]):
                for i in range(N):
                    A_part[j,i] = np.float64(f2.readline())
            comm.Send([A_part, rcounts_M[k]*N, MPI.DOUBLE], dest=k, tag=0)
    A_part = np.empty((M_part, N), dtype=np.float64)
else:
    A_part = np.empty((M_part, N), dtype=np.float64)
    comm.Recv([A_part, M_part*N, MPI.DOUBLE], source=0, tag=0, status=None)  

if rank == 0:
    b = np.empty(M, dtype=np.float64)
    with open('Example-03-1_bData.dat', 'r') as f3:
        for j in range(M):
            b[j] = np.float64(f3.readline())
else:
    b = None

b_part = np.empty(M_part, dtype=np.float64)

comm.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE], 
              [b_part, M_part, MPI.DOUBLE], root=0)

if rank == 0:
    x = np.zeros(N, dtype=np.float64)
else:
    x = None

x_part = np.empty(rcounts_N[rank], dtype=np.float64)

comm.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE], 
              [x_part, rcounts_N[rank], MPI.DOUBLE], root=0)

x_part = conjugate_gradient_method(A_part, b_part, x_part, 
                                   N, rcounts_N[rank], rcounts_N, displs_N)

comm.Gatherv([x_part, rcounts_N[rank], MPI.DOUBLE], 
             [x, rcounts_N, displs_N, MPI.DOUBLE], root=0)

if rank == 0:
    plt.style.use('dark_background')
    fig = plt.figure()
    ax = plt.axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i'); ax.set_ylabel('x[i]')
    ax.plot(np.arange(N), x, '-y', lw=3)
    plt.show()

#------------------------------------------------------------------
