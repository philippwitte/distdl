from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

n = 2
m = 4

A = np.ones((n, m))*(rank+1)

#######################################################################
# Forward 

# Reduce
if rank == 0:
    B = np.empty((n, m))
else:
    B = None
comm.Reduce(A, B, op=MPI.SUM, root=0)

# Scatter
C = np.empty((n, m // size))
comm.Scatter(B, C, root=0)

#######################################################################
# Backward

# Gather
if rank == 0:
    D = np.empty((n, m))
else:
    D = None
comm.Gather(C, D, root=0)

if rank != 0:
    D = np.empty((n, m))
comm.Bcast(D, root=0)
print(D)


[1, 1, 1, 1] -> [4, 1, 1, 1]