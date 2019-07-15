# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

def compute_size(model):
    """
    Size of model (in MB).
    """

    res = 0
    for n, p in model.named_parameters():
        res += p.numel()

    return res * 4 / 1024 / 1024
