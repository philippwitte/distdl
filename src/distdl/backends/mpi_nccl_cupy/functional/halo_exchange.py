__all__ = ["HaloExchangeFunction"]

import numpy as np
import cupy as cp
import torch
from mpi4py import MPI

from distdl.utilities.slicing import compute_nd_slice_shape
from distdl.utilities.torch import zero_volume_tensor


class HaloExchangeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, P_x, slices, buffers, neighbor_ranks):

        device = input.device
        ctx.slices = slices
        ctx.buffers = buffers
        ctx.neighbor_ranks = neighbor_ranks
        ctx.P_x = P_x
        ctx.device = device

        if not P_x.active:
            return zero_volume_tensor(input.shape[0], device=device)

        ctx.mark_dirty(input)

        if P_x.size == 1:
            return input

        dim = P_x.dim
        for i in range(dim):

            lbs, lgs, rbs, rgs = slices[i]
            lbb, lgb, rbb, rgb = buffers[i]
            if lbb is not None:
                lbb = lbb.get_view(compute_nd_slice_shape(lbs))
            if lgb is not None:
                lgb = lgb.get_view(compute_nd_slice_shape(lgs))
            if rbb is not None:
                rbb = rbb.get_view(compute_nd_slice_shape(rbs))
            if rgb is not None:
                rgb = rgb.get_view(compute_nd_slice_shape(rgs))
            lrank, rrank = neighbor_ranks[i]

            if lbb is not None:
                cp.copyto(lbb, cp.asarray(input.detach()[lbs]))
            if rbb is not None:
                cp.copyto(rbb, cp.asarray(input.detach()[rbs]))

            ltag = 0
            rtag = 1

            # Communication
            cp.cuda.nccl.groupStart()
            if lgb is not None:
                #stream_lgb = cp.cuda.Stream(non_blocking=True)
                P_x._nccl.recv(lgb, lrank, stream=None)
                event_lgb = cp.cuda.Event()
                event_lgb.record()
            if rgb is not None:
                #stream_rgb = cp.cuda.Stream(non_blocking=True)
                P_x._nccl.recv(rgb, rrank, stream=None)
                event_rgb = cp.cuda.Event()
                event_rgb.record()
            if lbb is not None:
                #stream_lbb = cp.cuda.Stream(non_blocking=True)
                P_x._nccl.send(lbb, lrank, stream=None)
            if rbb is not None:
                #stream_rbb = cp.cuda.Stream(non_blocking=True)
                P_x._nccl.send(rbb, rrank, stream=None)
            cp.cuda.nccl.groupEnd()

            # Wait for receive calls to complete
            if rgb is not None:
                cp.cuda.runtime.eventSynchronize(event_rgb.ptr)
                input[rgs] = torch.as_tensor(rgb, device=device)

            if lgb is not None:
                cp.cuda.runtime.eventSynchronize(event_lgb.ptr)
                input[lgs] = torch.as_tensor(lgb, device=device)

        return input

    @staticmethod
    def backward(ctx, grad_output):

        slices = ctx.slices
        buffers = ctx.buffers
        neighbor_ranks = ctx.neighbor_ranks
        P_x = ctx.P_x
        device = ctx.device

        assert grad_output.device == device

        if not P_x.active:
            return zero_volume_tensor(grad_output.shape[0], device=device), None, None, None, None

        if P_x.size == 1:
            return grad_output, None, None, None, None

        ctx.mark_dirty(grad_output)

        dim = P_x.dim
        for i in reversed(range(dim)):

            lbs, lgs, rbs, rgs = slices[i]
            lbb, lgb, rbb, rgb = buffers[i]
            if lbb is not None:
                lbb = lbb.get_view(compute_nd_slice_shape(lbs))
            if lgb is not None:
                lgb = lgb.get_view(compute_nd_slice_shape(lgs))
            if rbb is not None:
                rbb = rbb.get_view(compute_nd_slice_shape(rbs))
            if rgb is not None:
                rgb = rgb.get_view(compute_nd_slice_shape(rgs))
            lrank, rrank = neighbor_ranks[i]

            if lgb is not None:
                cp.copyto(lgb, cp.asarray(grad_output.detach()[lgs]))
                grad_output[lgs] = 0.0
            if rgb is not None:
                cp.copyto(rgb, cp.asarray(grad_output.detach()[rgs]))
                grad_output[rgs] = 0.0

            ltag = 0
            rtag = 1

            cp.cuda.nccl.groupStart()
            if lbb is not None:
                P_x._nccl.recv(lbb, lrank, stream=None)
                event_lbb = cp.cuda.Event()
                event_lbb.record()
            if rbb is not None:
                P_x._nccl.recv(rbb, rrank, stream=None)
                event_rbb = cp.cuda.Event()
                event_rbb.record()
            if lgb is not None:
                P_x._nccl.send(lgb, lrank, stream=None)
            if rgb is not None:
                P_x._nccl.send(rgb, rrank, stream=None)
            cp.cuda.nccl.groupEnd()

            # Wait for receive calls to complete
            if lbb is not None:
                cp.cuda.runtime.eventSynchronize(event_lbb.ptr)
                grad_output[lbs] += torch.as_tensor(lbb, device=device)

            if rbb is not None:
                cp.cuda.runtime.eventSynchronize(event_rbb.ptr)
                grad_output[rbs] += torch.as_tensor(rbb, device=device)

        return grad_output, None, None, None, None
