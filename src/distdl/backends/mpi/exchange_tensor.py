import numpy as np
from mpi4py import MPI


# Holy cow this is a touchy function, be very careful if modifying it...
def exchange_tensor_structure(tensor, P_send, P_recv):

    if not P_send.active and not P_recv.active:
        return None, None, None

    requests = []

    if P_send.active:
        # We have the values, set up copies of them to work with
        rg_int_send = np.array([-1], dtype=np.int)
        tensor_dim_send = np.array([len(tensor.shape)], dtype=np.int)
        tensor_sizes_send = np.array(tensor.shape, dtype=np.int)

        # Need to send non-Python types, so convert the boolean temporarily
        rg_int_send[0] = 1 if tensor.requires_grad else 0
        req = P_send.comm.Iallreduce(MPI.IN_PLACE, rg_int_send, op=MPI.MAX)
        requests.append(req)

        # Sending processes know the shape, so they can send a copy of the
        # data.  We will ignore this variable later.
        req = P_send.comm.Iallreduce(MPI.IN_PLACE, tensor_dim_send, op=MPI.MAX)
        requests.append(req)

        # Similarly, sending processes know the tensor shape, so they can send
        # a copy of it, but we will not use that copy for our actual return
        # value.
        req = P_send.comm.Iallreduce(MPI.IN_PLACE, tensor_sizes_send, op=MPI.MAX)
        requests.append(req)

        # This is safe to do before the requests complete, none of these are
        # part of the communcation structure
        tensor_requires_grad = tensor.requires_grad
        tensor_dim = len(tensor.shape)
        tensor_sizes = np.array(tensor.shape, dtype=np.int)

    # If the process is a receiving process, but doesn't already know the data
    # because it is the _same_ sending process, then we receive the results.
    # If it is a receiving process that sent data to a different set of
    # processes, we still have to complete the receive, even though later we
    # will not use that data.
    if (P_send != P_recv) and P_recv.active:
        # We have the values, set up copies of them to work with
        rg_int_recv = np.array([-1], dtype=np.int)
        tensor_dim_recv = np.array([-1], dtype=np.int)

        # Everyone needs to receive this, but we don't need it for future
        # communication in this function so we can defer receiving the data.
        req = P_recv.comm.Iallreduce(MPI.IN_PLACE, rg_int_recv, op=MPI.MAX)
        requests.append(req)

        # We need this value for the next communication, so we have to wait
        # for it to complete before moving on.
        req = P_recv.comm.Iallreduce(MPI.IN_PLACE, tensor_dim_recv, op=MPI.MAX)
        req.Wait()

        tensor_sizes_recv = np.zeros(tensor_dim_recv, dtype=np.int)
        tensor_sizes_recv[:] = -1
        req = P_recv.comm.Iallreduce(MPI.IN_PLACE, tensor_sizes_recv, op=MPI.MAX)
        requests.append(req)

    # Make sure all requests, including the final recv all reduce complete
    # before receiving processes can actually copy the data out.
    MPI.Request.Waitall(requests)

    # Wait until the communication is complete to set these values.  Only
    # receiving ranks that do not have the data originally should enter here.
    if P_recv.active and not P_send.active:
        tensor_requires_grad = bool(rg_int_recv[0] == 1)
        tensor_dim = tensor_dim_recv[0]
        tensor_sizes = tensor_sizes_recv

    # Finally, everyone should have valid data.  Any sending rank created it
    # from input data.  Any receving _only_ rank used what it was given.
    return tensor_requires_grad, tensor_dim, tensor_sizes
