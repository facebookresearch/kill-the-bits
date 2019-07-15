# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch.optim.optimizer import Optimizer, required

from utils.reshape import reshape_weight, reshape_back_weight


class CentroidSGD(Optimizer):
    """
    Performs centroids finetuning given the block assignments.

    Args:
        - params: model.parameters()
        - assignments: assignments of each block of size n_blocks
          in the reshaped + unrolled weight matrix of the layers
        - n_centroids: number of centroids used to quantized the layer
        - n_blocks: number of blocks in the reshaped weight matrix
        - lr, momentum, dampening, weight_decay, nesterov: classical
          optimizer parameters, see PyTorch's documentation

    Remarks:
        - After each iteration, the gradients corresponding to the blokcs
          assigned to centroid k are averaged and the same update using
          this averaged gradient is applied to all the corresponding blocks
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(CentroidSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CentroidSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """
        Performs a single optimization step on the centroids.

        Args:
            - closure (callable, optional): A closure that reevaluates the model
               and returns the loss.

        Remarks:
            - The "reduce gradients" step is equivalent to (but 2x faster than) the following lines:
              ```
              for k in range(n_centroids):
                  mean_k = d_p_unroll[:, assignments == k].mean(dim=1, keepdim=True)
                  d_p_unroll[:, assignments == k] = mean_k
             ```
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            assignments = group['assignments']
            kernel_size = group['kernel_size']
            n_centroids = group['n_centroids']
            n_blocks = group['n_blocks']

            for p in group['params']:
                # recover gradient
                if p.grad is None:
                    continue

                # unroll gradients
                d_p = p.grad.data
                d_p_unroll = reshape_weight(d_p)
                d_p_unroll = torch.cat(d_p_unroll.chunk(n_blocks, dim=0), dim=1)

                # reduce gradients
                select = assignments[:, None] == torch.arange(int(n_centroids), device=p.device)
                select = select.float() / torch.bincount(assignments).float()
                d_p_unroll = d_p_unroll.mm(select)[:, assignments]

                # roll gradients back
                conv = len(p.size()) == 4
                d_p = torch.cat(d_p_unroll.chunk(n_blocks, dim=1), dim=0)
                d_p = reshape_back_weight(d_p, k=kernel_size, conv=conv)

                # handle weight decay and momentum
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # SGD update
                p.data.add_(-group['lr'], d_p)

        return loss
