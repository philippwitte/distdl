# This example demonstrates the behavior of the all-reduce primitive.
#
# It requires 6 workers to run.
#
# Run with, e.g.,
#     > mpirun -np 6 python ex_all_reduce.py

import numpy as np
import torch
from mpi4py import MPI
from distdl.backend import BackendProtocol, FrontEndProtocol, ModelProtocol, init_distdl

import distdl.utilities.slicing as slicing
from distdl.backends.common.partition import MPIPartition
from distdl.nn.reduce_scatter import ReduceScatter
from distdl.utilities.torch import zero_volume_tensor

init_distdl(frontend_protocol=FrontEndProtocol.MPI,
            backend_protocol=BackendProtocol.MPI,
            model_protocol=ModelProtocol.NUMPY)

# Set up MPI cartesian communicator
P_world = MPIPartition(MPI.COMM_WORLD)
P_world._comm.Barrier()

# On the assumption of 1-to-1 mapping between ranks and GPUs
# torch.cuda.set_device(P_world.rank % torch.cuda.device_count())
# P_world.device = torch.cuda.current_device()
# P_world.device = torch.device("cpu")

# Create the input/output partition (using the first worker)
in_shape = (2, 3)
in_size = np.prod(in_shape)
in_workers = np.arange(0, in_size)

P_x_base = P_world.create_partition_inclusive(in_workers)
P_x = P_x_base.create_cartesian_topology_partition(in_shape)

# The all-reduce layer operates along partitiond dimensions, not tensor
# dimensions.  Thus, along the dimensions that the reduction applies, the
# subtensors all must be the same size.  Thus, this global shape is evenly
# divisible by the partition.  Later we will have an example for applying the
# reduction on the tensor itself.
x_global_shape = np.array([6, 18])

# Setup the input tensor.  Any worker in P_x will generate its part of the
# input tensor.  Any worker not in P_x will have a zero-volume tensor.
#
# Input tensor will be (on a 2 x 3 partition):
# [ [ 1 1 | 2 2 | 3 3 ]
#   [ 1 1 | 2 2 | 3 3 ]
#   [ 1 1 | 2 2 | 3 3 ]
#   -------------------
#   [ 4 4 | 5 5 | 6 6 ]
#   [ 4 4 | 5 5 | 6 6 ]
#   [ 4 4 | 5 5 | 6 6 ] ]

x = zero_volume_tensor(device=P_x.device)

if P_x.active:
    x_local_shape = slicing.compute_subshape(P_x.shape,
                                             P_x.index,
                                             x_global_shape)
    x = torch.zeros(*x_local_shape, device=x.device) + (P_x.rank + 1)

x.requires_grad = True

print(f"P_world.rank {P_world.rank}; P_x.index {P_x.index}; x value: \n{x}\n")

# Create the all-reduce layer.  Note, only one of the keep/reduce axes is
# required.  If they are both specified they must be mutually coherent.
# Commented out declarations are equivalent.  `axes_reduce` is equivalent
# to PyTorch's dimension argument in torch.sum().
#
# Here we reduce the columns (axis 1), along the rows.
all_reduce_cols = ReduceScatter(P_x, axes_reduce_scatter=(1,))
# all_reduce_cols = AllSumReduce(P_x, axes_keep=(0,))
# all_reduce_cols = AllSumReduce(P_x, axes_reduce=(1,), axes_keep=(0,))
#
# Output tensor will be (on a 2 x 3 partition):
# [ [  6  6 |  6  6 |  6  6 ]
#   [  6  6 |  6  6 |  6  6 ]
#   [  6  6 |  6  6 |  6  6 ]
#   -------------------------
#   [ 15 15 | 15 15 | 15 15 ]
#   [ 15 15 | 15 15 | 15 15 ]
#   [ 15 15 | 15 15 | 15 15 ] ]
#print(x.shape)
y = all_reduce_cols(x)

print(y)