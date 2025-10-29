import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

def u_init(x):
    u_init = np.sin(3*np.pi*(x - 1/6))
    return u_init

def u_left(t):
    u_left = -1.
    return u_left

def u_right(t):
    u_right = 1.
    return u_right

start_time = MPI.Wtime()

a = 0.0; b = 1.0
t_0 = 0.0; T = 6.0

eps = 10**(-1.5)

N = 800
M = 300_000

h = (b - a) / N
tau = (T - t_0) / M

x = np.linspace(a, b, N+1)
t = np.linspace(t_0, T, M+1)

if rank == 0:
    ave, res = divmod(N + 1, numprocs)
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs = np.empty(numprocs, dtype=np.int32)
    for k in range(0, numprocs):
        if k < res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        if k == 0:
            displs[k] = 0
        else:
            displs[k] = displs[k-1] + rcounts[k-1]   
else:
    rcounts = None; displs = None

N_part = np.array(0, dtype=np.int32)

comm.Scatter([rcounts, 1, MPI.INT],
             [N_part, 1, MPI.INT], root=0)

if rank == 0:
    rcounts_from_0 = np.empty(numprocs, dtype=np.int32)
    displs_from_0 = np.empty(numprocs, dtype=np.int32)
    rcounts_from_0[0] = rcounts[0] + 1
    displs_from_0[0] = 0
    for k in range(1, numprocs-1):
        rcounts_from_0[k] = rcounts[k] + 2
        displs_from_0[k] = displs[k] - 1
    rcounts_from_0[numprocs-1] = rcounts[numprocs-1] + 1  
    displs_from_0[numprocs-1] = displs[numprocs-1] - 1
else:
    rcounts_from_0 = None; displs_from_0 = None

N_part_aux = np.array(0, dtype=np.int32)

comm.Scatter([rcounts_from_0, 1, MPI.INT],
             [N_part_aux, 1, MPI.INT], root=0)

if rank == 0:
    u = np.empty((M+1, N+1), dtype=np.float64)
    for n in range(N + 1):
        u[0, n] = u_init(x[n]) 
else:
    u = np.empty((M+1, 0), dtype=np.float64)

u_part = np.empty(N_part, dtype=np.float64)
u_part_aux = np.empty(N_part_aux, dtype=np.float64)

for m in range(M):
    comm.Scatterv([u[m], rcounts_from_0, displs_from_0, MPI.DOUBLE], 
                  [u_part_aux, N_part_aux, MPI.DOUBLE], root=0)

    for n in range(1, N_part_aux - 1):
        u_part[n-1] = u_part_aux[n] + \
            eps*tau/h**2*(u_part_aux[n+1] - 2*u_part_aux[n] + u_part_aux[n-1]) + \
                tau/(2*h)*u_part_aux[n]*(u_part_aux[n+1] - u_part_aux[n-1]) + \
                    tau*u_part_aux[n]**3

    if rank == 0:
        u_part = np.hstack((np.array(u_left(t[m+1]), dtype=np.float64), u_part[0:N_part-1]))
    elif rank == numprocs-1:
        u_part = np.hstack((u_part[0:N_part-1], np.array(u_right(t[m+1]), dtype=np.float64)))

    comm.Gatherv([u_part, N_part, MPI.DOUBLE], 
                 [u[m+1], rcounts, displs, MPI.DOUBLE], root=0)

end_time = MPI.Wtime()

if rank == 0:
    print('N={}, M={}'.format(N, M))
    print('Number of MPI process is {}'.format(numprocs))
    print('Elapsed time is {:.4f} sec.'.format(end_time-start_time))

    np.savez('Example-08-1_Results', x=x, u=u)
