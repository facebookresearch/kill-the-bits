# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn.functional as F


def reshape_weight(weight):
    """
    C_out x C_in x k x k -> (C_in x k x k) x C_out.
    """

    if len(weight.size()) == 4:
        C_out, C_in, k, k = weight.size()
        return weight.view(C_out, C_in * k * k).t()
    else:
        return weight.t()


def reshape_back_weight(weight, k=3, conv=True):
    """
    (C_in x k x k) x C_out -> C_out x C_in x k x k.
    """

    if conv:
        C_in_, C_out = weight.size()
        C_in = C_in_ // (k * k)
        return weight.t().view(C_out, C_in, k, k).contiguous()
    else:
        return weight.t().contiguous()


def reshape_activations(activations, k=3, stride=(1, 1), padding=(1, 1), groups=1):
    """
    N x C_in x H x W -> (N x H x W) x C_in.
    """

    if len(activations[0].size()) == 4:
        # gather activations
        a_stacked = []

        for n in range(len(activations)):
            a_padded = F.pad(activations[n], (padding[1], padding[1], padding[0], padding[0]))
            N, C, H, W = a_padded.size()
            for i in range(0, H - k + 1, stride[0]):
                for j in range(0, W - k + 1, stride[1]):
                    a_stacked.append(a_padded[:, :, i:i + k, j:j + k])

        # reshape according to weight
        a_reshaped = reshape_weight(torch.cat(a_stacked, dim=0)).t()

        # group convolutions (e.g. depthwise convolutions)
        a_reshaped_groups = torch.cat(a_reshaped.chunk(groups, dim=1), dim=0)

        return a_reshaped_groups

    else:
        return torch.cat(activations, dim=0)
