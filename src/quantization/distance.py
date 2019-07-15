# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from functools import reduce

import torch
import torch.nn as nn


class ComputeDistances(nn.Module):
    """ Computes distances as described in the file em.py using the map/reduce paradigm.

    Args:
        - M: weight matrix
        - centroids: centroids used to compute the distance

    Remarks:
        - We split the input activations into up to 8 GPUs by relying on DataParallel.
        - The computation of distances is done per GPU with its chunk of activations (map)
        - The distances are then aggregated (reduce)

    """
    def __init__(self, M, centroids):
        super(ComputeDistances, self).__init__()
        self.distances = nn.parallel.DataParallel(Distances(M, centroids)).cuda()
        self.reduce = Reduce()

    def forward(self, in_activations):
        return self.reduce(self.distances(in_activations))

    def update_centroids(self, centroids):
        self.distances.module.centroids.data = centroids


class Distances(nn.Module):
    """
    Computes distances using broadcasting (map step). This layer automatically chunks the
    centroids and the weight matrix M so that the computation fits into the GPU

    Remarks:
        - The dimensions of the centroids and the weight matrix must be "chunkable enough" since
          we divide them by two until ths computation fits on the GPU
        - For debuging purposes, we advise the programmer to use only one GPU by setting
          CUDA_VISIBLE_DEVICES=1
    """

    def __init__(self, M, centroids):
        super(Distances, self).__init__()
        self.M = nn.Parameter(M, requires_grad=False)
        self.centroids = nn.Parameter(centroids, requires_grad=False)

    def forward(self, in_activations):
        # two cases
        nb_M_chunks = 1
        nb_centroids_chunks = 1

        while True:
            try:
                return torch.cat([
                            torch.cat([
                                torch.matmul(
                                    in_activations[None, :, :],
                                    M_c[None, :, :] - centroids_c[:, :, None]
                                ).norm(p=2, dim=1).pow(2)
                                for centroids_c in self.centroids.chunk(nb_centroids_chunks, dim=0)
                            ], dim=0)
                            for M_c in self.M.chunk(nb_M_chunks, dim=1)
                        ], dim=1)
            except RuntimeError:
                nb_M_chunks *= 2
                nb_centroids_chunks *= 2


class Reduce(nn.Module):
    """
    Reduces the distances per GPU accordingly (reduce step)

    Remarks:
        - Reduction is performed on GPU0 by default, which may lead to memory errors with too large
          a mode or a batch size. For debuging purposes, we advise the programmer to use only one GPU by setting CUDA_VISIBLE_DEVICES=1
    """

    def __init__(self):
        super(Reduce, self).__init__()
        self.n_gpus = torch.cuda.device_count()

    def forward(self, distances):
        return reduce(lambda x, y: x + y, distances.chunk(self.n_gpus, dim=0)).sqrt()
