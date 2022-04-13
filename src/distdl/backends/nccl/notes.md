# MPI4PY

# Attributes
comm    # communicator
group   # group

# Group functions
group.free  # releast group
group.Incl(ranks)  # new group from ranks
MPI.Group.Union(group, other_group) # combine two groups
check_identical_group(group1, group2)   # check that two groups are identical

# Communicator functions
comm.Get_rank()             # Rank
comm.Get_size()             # Size
comm.Free()                 # Release comm
comm.Create_group(group)    # create comm from group
comm.Get_group()
check_identical_comm(comm1, comm2)  # check that two communicators are identical
comm.Create_cart(shape)     # cartesian communicator
comm.Get_cart_rank(index)          # Get rank of cartesian communicator

comm.AllGather
comm.Bcast
comm.Allreduce
comm.Sub    # MPI_Comm_sub
comm.Get_coords(rank)

# DistDL Partition
MPIPartition(comm, group, root=None)



###############################################################################################
# Pytorch

