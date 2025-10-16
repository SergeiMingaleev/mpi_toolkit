from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

def conjugate_gradient_method(A_part, b_part, x_part, N_part, M_part, 
                              N, comm, comm_row, comm_col, rank):
        
    r_part = np.empty(N_part, dtype=np.float64)
    p_part = np.empty(N_part, dtype=np.float64)
    q_part = np.empty(N_part, dtype=np.float64)
    
    if rank in range(num_col):
        ScalP = np.array(0, dtype=np.float64)
        ScalP_temp = np.empty(1, dtype=np.float64)
    
    s = 1
    
    p_part = np.zeros(N_part, dtype=np.float64)

    while s <= N:

        if s == 1:
            comm_col.Bcast([x_part, N_part, MPI.DOUBLE], root=0)
            Ax_part_temp = np.dot(A_part, x_part)
            Ax_part = np.empty(M_part, dtype=np.float64)
            comm_row.Reduce([Ax_part_temp, M_part, MPI.DOUBLE], 
                            [Ax_part, M_part, MPI.DOUBLE], 
                            op=MPI.SUM, root=0)
            if rank in range(0, numprocs, num_col):
                b_part = Ax_part - b_part
            comm_row.Bcast([b_part, M_part, MPI.DOUBLE], root=0)    
            r_part_temp = np.dot(A_part.T, b_part)
            comm_col.Reduce([r_part_temp, N_part, MPI.DOUBLE], 
                            [r_part, N_part, MPI.DOUBLE], 
                            op=MPI.SUM, root=0)
        else:
            if rank in range(num_col):
                ScalP_temp[0] = np.dot(p_part, q_part)
                comm_row.Allreduce([ScalP_temp, 1, MPI.DOUBLE], 
                                   [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
                r_part = r_part - q_part/ScalP
        
        if rank in range(num_col):
            ScalP_temp[0] = np.dot(r_part, r_part)
            comm_row.Allreduce([ScalP_temp, 1, MPI.DOUBLE], 
                               [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
            p_part = p_part + r_part/ScalP
        
        comm_col.Bcast([p_part, N_part, MPI.DOUBLE], root=0)
        Ap_part_temp = np.dot(A_part, p_part)
        Ap_part = np.empty(M_part, dtype=np.float64)
        comm_row.Allreduce([Ap_part_temp, M_part, MPI.DOUBLE], 
                           [Ap_part, M_part, MPI.DOUBLE], op=MPI.SUM)
        q_part_temp = np.dot(A_part.T, Ap_part)
        comm_col.Reduce([q_part_temp, N_part, MPI.DOUBLE], 
                        [q_part, N_part, MPI.DOUBLE], 
                        op=MPI.SUM, root=0)
        
        if rank in range(num_col):
            ScalP_temp[0] = np.dot(p_part, q_part)
            comm_row.Allreduce([ScalP_temp, 1, MPI.DOUBLE], 
                               [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
            x_part = x_part - p_part/ScalP
        
        s = s + 1
    
    return x_part

if rank == 0:
    with open('Example-03_in.dat', 'r') as f1:
        M = np.array(np.int32(f1.readline()))
        N = np.array(np.int32(f1.readline()))
else:
    N = np.array(0, dtype=np.int32)

comm.Bcast([N, 1, MPI.INT], root=0)

num_col = num_row = np.int32(np.sqrt(numprocs))

def auxiliary_arrays_determination(M, num):
    ave, res = divmod(M, num)
    rcounts = np.empty(num, dtype=np.int32)
    displs = np.empty(num, dtype=np.int32)
    for k in range(0, num):
        if k < res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        if k == 0:
            displs[k] = 0
        else:
            displs[k] = displs[k-1] + rcounts[k-1]   
    return rcounts, displs

if rank == 0:
    rcounts_M, displs_M = auxiliary_arrays_determination(M, num_row)
    rcounts_N, displs_N = auxiliary_arrays_determination(N, num_col)
else:
    rcounts_M = None; displs_M = None
    rcounts_N = None; displs_N = None

M_part = np.array(0, dtype=np.int32); N_part = np.array(0, dtype=np.int32)

comm_col = comm.Split(rank % num_col, rank)
comm_row = comm.Split(rank // num_col, rank)

if rank in range(num_col):
    comm_row.Scatter([rcounts_N, 1, MPI.INT], 
                     [N_part, 1, MPI.INT], root=0) 
if rank in range(0, numprocs, num_col):
    comm_col.Scatter([rcounts_M, 1, MPI.INT],
                     [M_part, 1, MPI.INT], root=0) 

comm_col.Bcast([N_part, 1, MPI.INT], root=0)
comm_row.Bcast([M_part, 1, MPI.INT], root=0)  

A_part = np.empty((M_part, N_part), dtype=np.float64)

group = comm.Get_group()

if rank == 0:
    with open('Example-03_AData.dat', 'r') as f2:
        for m in range(num_row):
            a_temp = np.empty(rcounts_M[m]*N, dtype=np.float64)
            for j in range(rcounts_M[m]):
                for n in range(num_col):
                    for i in range(rcounts_N[n]):
                        a_temp[rcounts_M[m]*displs_N[n] + j*rcounts_N[n] + i] = np.float64(f2.readline())
            if m == 0:
                comm_row.Scatterv([a_temp, rcounts_M[m]*rcounts_N, rcounts_M[m]*displs_N, MPI.DOUBLE],
                                  [A_part, M_part*N_part, MPI.DOUBLE], root=0)
            else:
                group_temp = group.Range_incl([(0,0,1), (m*num_col,(m+1)*num_col-1,1)])
                comm_temp = comm.Create(group_temp)
                rcounts_N_temp = np.hstack((np.array(0, dtype=np.int32), rcounts_N))
                displs_N_temp = np.hstack((np.array(0, dtype=np.int32), displs_N))
                comm_temp.Scatterv([a_temp, rcounts_M[m]*rcounts_N_temp, rcounts_M[m]*displs_N_temp, MPI.DOUBLE],
                                   [np.empty(0, dtype=np.float64), 0, MPI.DOUBLE], root=0)
                group_temp.Free(); comm_temp.Free()
else:
    if rank in range(num_col):
        comm_row.Scatterv([None, None, None, None], 
                          [A_part, M_part*N_part, MPI.DOUBLE], root=0)
    for m in range(1, num_row):
        group_temp = group.Range_incl([(0,0,1), (m*num_col,(m+1)*num_col-1,1)])
        comm_temp = comm.Create(group_temp)
        if rank in range(m*num_col, (m+1)*num_col):
            comm_temp.Scatterv([None, None, None, None], 
                               [A_part, M_part*N_part, MPI.DOUBLE], root=0)
            comm_temp.Free()
        group_temp.Free()
    
if rank == 0:
    b = np.empty(M, dtype=np.float64)
    with open('Example-03_bData.dat', 'r') as f3:
        for j in range(M):
            b[j] = np.float64(f3.readline())
else:
    b = None
    
b_part = np.empty(M_part, dtype=np.float64) 
 	
if rank in range(0, numprocs, num_col):
    comm_col.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE], 
                      [b_part, M_part, MPI.DOUBLE], root=0)

if rank == 0:
    x = np.zeros(N, dtype=np.float64)
else:
    x = None
    
x_part = np.empty(N_part, dtype=np.float64) 

if rank in range(num_col):
    comm_row.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE], 
                    [x_part, N_part, MPI.DOUBLE], root=0)

x_part = conjugate_gradient_method(A_part, b_part, x_part, N_part, M_part, 
                                   N, comm, comm_row, comm_col, rank)

if rank in range(num_col):
    comm_row.Gatherv([x_part, N_part, MPI.DOUBLE], 
                     [x, rcounts_N, displs_N, MPI.DOUBLE], root=0)

if rank == 0:
    import matplotlib.pyplot as plt

    plt.style.use('dark_background')
    fig = plt.figure()
    ax = plt.axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i'); ax.set_ylabel('x[i]')
    ax.plot(np.arange(N), x, '-y', lw=3)
    plt.show()
