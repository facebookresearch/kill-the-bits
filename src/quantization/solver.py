# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def solve_stack(A, B, A_pinv=None):
    """
    Finds the minimizer X of ||A[X, X, ..., X] - B||^2
    using the pseudo-inverse of A.

    Args:
        - A: weight matrix of size (n x p)
        - B: bias matrix of size (n x q)
        - A_pinv: the pseudo-inverse of A of size (p x n) (optional)

    Remarks:
        - The pseudo-inverse of A can be passed as an argument to factor
          computations when solve_stack is called many times with the same
          matrix A
        - The unknown X is of size (p x 1). Here, [X, X, ..., X] denotes the
          column vector X stacked q times horizontally
    """

    A_pinv = torch.pinverse(A) if A_pinv is None else A_pinv
    return torch.matmul(A_pinv, B.sum(dim=1)) / B.size(1)
