# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from .reshape import reshape_weight, reshape_back_weight


def centroids_from_weights(M, assignments, n_centroids, n_blocks):
    """
    Recovers the centroids from an already quantized matrix M.
    Args:
        - M: already quantized matrix
        - assignments: size (n_vectors)
        - n_centroids: number of centroids
        - n_blocks: niumber of blocks per column
    Remarks:
        - This function assumes that all the clusters are non-empty
        - This function consists in two steps:
            (1) reshape the 2D/4D matrix (whether fully-connected or convolutional) into a 2D matrix
            (2) unroll the obtained 2D matrix according to the number of blocks
    """

    M_reshaped = reshape_weight(M)
    M_unrolled = torch.cat(M_reshaped.chunk(n_blocks, dim=0), dim=1)
    size_block = M_unrolled.size(0)
    centroids = torch.zeros(n_centroids, size_block, device=M.device)
    for k in range(n_centroids):
        centroids[k] = M_unrolled[:, assignments == k][:, 0]

    return centroids


def weight_from_centroids(centroids, assignments, n_blocks, k, conv):
    """
    Constructs the 2D matrix from its centroids.
    Args:
        - centroids: size (block_size x n_centroids)
        - assignments: size (n_vectors)
        _ n_blocks: numnber of blocks per column
        - k: kernel size (set to 1 if not is_conv)
        - is_conv: convolutional or linear layer
    Remarks:
        - This function consists in two steps:
            (1) get the 2D unrolled weight matrix
            (2) reshape it in the case of fully-connected of convolutional layer
    """

    M_hat_unrolled = torch.cat(centroids[assignments].t().chunk(n_blocks, dim=1), dim=0)
    M_hat_reshaped = reshape_back_weight(M_hat_unrolled, k=k, conv=conv)

    return M_hat_reshaped
