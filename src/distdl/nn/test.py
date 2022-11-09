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

B = np.empty((n, m // size))
comm.Ireduce_scatter(A, B, op=MPI.SUM)
#print(B)

#######################################################################
# Backward

# Allgather
C = np.empty((n, m))
comm.Allgather(B, C)

print(C)