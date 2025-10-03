from mpi4py import MPI
from numpy import empty, array, int32, float64, ones, dot

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

if rank == 0 :
    f1 = open('Example-01-1_in.dat', 'r')
    N = array(int32(f1.readline()))
    M = array(int32(f1.readline()))
    f1.close()
else :
    N = array(0, dtype=int32)

comm.Bcast([N, 1, MPI.INT], root=0)

if rank == 0 :
    ave, res = divmod(M, numprocs-1)
    rcounts = empty(numprocs, dtype=int32)
    displs = empty(numprocs, dtype=int32)
    rcounts[0] = 0; displs[0] = 0
    for k in range(1, numprocs) : 
        if k < 1 + res :
            rcounts[k] = ave + 1
        else :
            rcounts[k] = ave
        displs[k] = displs[k-1] + rcounts[k-1]
else :
    rcounts = None; displs = None

M_part = array(0, dtype=int32)

comm.Scatterv([rcounts, ones(numprocs, dtype=int32), array(range(numprocs)), MPI.INT], 
              [M_part, 1, MPI.INT], root=0)  # comm.Scatter([rcounts, 1, MPI.INT], [M_part, 1, MPI.INT], root=0) 

A_part = empty((M_part, N), dtype=float64)  
if rank == 0 :
    f2 = open('Example-01-1_AData.dat', 'r')
    A = empty((M,N), dtype=float64)
    for j in range(M) :
        for i in range(N) :
            A[j,i] = float64(f2.readline())
    f2.close()
    comm.Scatterv([A, rcounts*N, displs*N, MPI.DOUBLE], 
                  [A_part, M_part*N, MPI.DOUBLE], root=0)   
else :
    comm.Scatterv([None, None, None, None], 
                  [A_part, M_part*N, MPI.DOUBLE], root=0)    
    
x = empty(N, dtype=float64)
if rank == 0 :
    f3 = open('Example-01-1_xData.dat', 'r')
    for i in range(N) :
        x[i] = float64(f3.readline())
    f3.close()
    
comm.Bcast([x, N, MPI.DOUBLE], root=0)

b_part = dot(A_part,x)
 	
if rank == 0 :
    b = empty(M, dtype=float64)
else:
    b = None

comm.Gatherv([b_part, M_part, MPI.DOUBLE], [b, rcounts, displs, MPI.DOUBLE], root=0)

if rank == 0:
# Сохраняем результат вычислений в файл
    f4 = open('Example-01-1_Results.dat', 'w')
    for j in range(M) :
        f4.write(str(b[j]))
        f4.write('\n')
    f4.close()
    
    print(b)
