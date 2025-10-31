import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

comm_cart = comm.Create_cart(dims=[numprocs], periods=[False], reorder=True)
rank_cart = comm_cart.Get_rank()

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

eps = 10**(-1.5)
a = 0.0; b = 1.0
t_0 = 0.0

T = 2.0
N = 800;    M = 100_000
# T = 0.02
# N = 8_000;  M = 100_000

h = (b - a) / N
tau = (T - t_0) / M

x = np.linspace(a, b, N+1)
t = np.linspace(t_0, T, M+1)

if rank_cart == 0:
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

comm_cart.Scatter([rcounts, 1, MPI.INT],
                  [N_part, 1, MPI.INT], root=0)

if rank_cart == 0:
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
displ_aux = np.array(0, dtype=np.int32)

comm_cart.Scatter([rcounts_from_0, 1, MPI.INT],
                  [N_part_aux, 1, MPI.INT], root=0)
comm_cart.Scatter([displs_from_0, 1, MPI.INT],
                  [displ_aux, 1, MPI.INT], root=0)

u_part_aux = np.empty((M + 1, N_part_aux), dtype=np.float64)

u_part_aux[0, :] = u_init(x[displ_aux:displ_aux+N_part_aux])

if rank_cart == 0:
    u_part_aux[1:, 0] = u_left(t[1:])
if rank_cart == numprocs-1:
    u_part_aux[1:, N_part_aux - 1] = u_right(t[1:])

u_part_aux_left_send = np.empty(1, dtype=np.float64)
u_part_aux_left_recv = np.empty(1, dtype=np.float64)
u_part_aux_right_send = np.empty(1, dtype=np.float64)
u_part_aux_right_recv = np.empty(1, dtype=np.float64)

# requests = [MPI.Request() for i in range(4)]
requests = [MPI.Prequest() for i in range(4)]

if rank_cart > 0:
    requests[0] = comm_cart.Send_init([u_part_aux_left_send, 1, MPI.DOUBLE],
                                      dest=rank_cart-1, tag=0)
    requests[1] = comm_cart.Recv_init([u_part_aux_left_recv, 1, MPI.DOUBLE],
                                      source=rank_cart-1, tag=0)
if rank_cart < numprocs-1:
    requests[2] = comm_cart.Send_init([u_part_aux_right_send, 1, MPI.DOUBLE],
                                      dest=rank_cart+1, tag=0)
    requests[3] = comm_cart.Recv_init([u_part_aux_right_recv, 1, MPI.DOUBLE],
                                      source=rank_cart+1, tag=0)

eps_tau_h2 = eps*tau/h**2
tau_2h = tau/(2*h)

for m in range(M):
    u_part_aux[m + 1, 1:-1] = u_part_aux[m, 1:-1] + \
        eps_tau_h2*(u_part_aux[m, 2:] - 2*u_part_aux[m, 1:-1] + u_part_aux[m, :-2]) + \
            tau_2h*u_part_aux[m, 1:-1]*(u_part_aux[m, 2:] - u_part_aux[m, :-2]) + \
                tau*u_part_aux[m, 1:-1]**3

    if rank_cart == 0:
        u_part_aux_right_send[0] = u_part_aux[m+1, N_part_aux-2]
        MPI.Prequest.Startall(requests[2:4])
        MPI.Request.Waitall(requests[2:4], statuses=None)
        u_part_aux[m+1, N_part_aux-1] = u_part_aux_right_recv[0]
    if rank_cart in range(1, numprocs-1):
        u_part_aux_left_send[0] = u_part_aux[m+1, 1]
        u_part_aux_right_send[0] = u_part_aux[m+1, N_part_aux-2]
        MPI.Prequest.Startall(requests)
        MPI.Request.Waitall(requests, statuses=None)
        u_part_aux[m+1, 0] = u_part_aux_left_recv[0]
        u_part_aux[m+1, N_part_aux-1] = u_part_aux_right_recv[0]
    if rank_cart == numprocs-1:
        u_part_aux_left_send[0] = u_part_aux[m+1, 1]
        MPI.Prequest.Startall(requests[0:2])
        MPI.Request.Waitall(requests[0:2], statuses=None)
        u_part_aux[m+1, 0] = u_part_aux_left_recv[0]

if rank_cart == 0:
    u_T = np.empty(N+1, dtype=np.float64)
else:
    u_T = None

if rank_cart == 0:
    comm_cart.Gatherv([u_part_aux[M, 0:N_part_aux-1], N_part, MPI.DOUBLE], 
                      [u_T, rcounts, displs, MPI.DOUBLE], root=0)
if rank_cart in range(1, numprocs-1):
    comm_cart.Gatherv([u_part_aux[M, 1:N_part_aux-1], N_part, MPI.DOUBLE], 
                      [u_T, rcounts, displs, MPI.DOUBLE], root=0)
if rank_cart == numprocs-1:
    comm_cart.Gatherv([u_part_aux[M, 1:N_part_aux], N_part, MPI.DOUBLE], 
                      [u_T, rcounts, displs, MPI.DOUBLE], root=0)

end_time = MPI.Wtime()

if rank_cart == 0:
    print('N={}, M={}'.format(N, M))
    print('Number of MPI process is {}'.format(numprocs))
    print('Elapsed time is {:.4f} sec.'.format(end_time-start_time))

    # Если нужно вывести график, закомментируйте `exit()` ниже:
    exit()

    from matplotlib import pyplot as plt
    plt.style.use('dark_background')
    fig = plt.figure()
    ax = plt.axes(xlim=(a,b), ylim=(-2.0, 2.0))
    ax.set_xlabel('x'); ax.set_ylabel('u')
    ax.plot(x,u_T, color='y', ls='-', lw=2)
    plt.show()
