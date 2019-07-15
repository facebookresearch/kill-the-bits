# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from operator import attrgetter

import torch
import torch.nn as nn

import torchvision.models.detection as models

import utils.utils_mask as utils

from data import load_data
from utils.training import evaluate
from utils.watcher import ActivationWatcher
from utils.utils import weight_from_centroids


parser = argparse.ArgumentParser(description='Inference for quantized networks')
parser.add_argument('--model', default='maskrcnn_resnet50_fpn', choices=['maskrcnn_resnet50_fpn'],
                    help='Model to use for inference')
parser.add_argument('--state-dict-compressed', default='', type=str,
                    help='Path to the compressed state dict of the model')
parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'],
                    help='For inference on CPU or on GPU')
parser.add_argument('--data-path', default='/datasets01/COCO/022719/', type=str,
                    help='Path to COCO dataset')
parser.add_argument('--batch-size', default=1, type=int,
                    help='Batch size for fiuetuning steps')
parser.add_argument('--n-workers', default=20, type=int,
                    help='Number of workers for data loading')
parser.add_argument('--aspect-ratio-group-factor', default=0, type=int,
                    help='Group ratio')


def main():
    # get arguments
    global args
    args = parser.parse_args()
    device = args.device

    state_dict_compressed = torch.load(args.state_dict_compressed)

    # instantiating model
    data_loader, data_loader_test, num_classes = load_data(args.data_path, batch_size=args.batch_size, nb_workers=args.n_workers,
        distributed=False, aspect_ratio_group_factor=args.aspect_ratio_group_factor)
    model = models.__dict__[args.model](pretrained=False).to(device)

    watcher = ActivationWatcher(model)
    compressed_layers = watcher.layers

    # non-compressed layers
    non_compressed_layers =  ['backbone.body.conv1', 'rpn.head.cls_logits', 'rpn.head.bbox_pred', 'roi_heads.mask_predictor.mask_fcn_logits']
    for layer in non_compressed_layers:
        compressed_layers.remove(layer)
        state_dict_layer = to_device(state_dict_compressed[layer], device)
        attrgetter(layer)(model).load_state_dict(state_dict_layer)
        attrgetter(layer)(model).float()

    for layer in compressed_layers:
        # recover centroids and assignments
        state_dict_layer = state_dict_compressed[layer]
        centroids = state_dict_layer['centroids'].float().to(device)
        assignments = state_dict_layer['assignments'].long().to(device)
        n_blocks = state_dict_layer['n_blocks']
        is_conv = state_dict_layer['is_conv']
        k = state_dict_layer['k']

        # instantiate matrix
        M_hat = weight_from_centroids(centroids, assignments, n_blocks, k, is_conv)
        attrgetter(layer + '.weight')(model).data = M_hat

    # batch norms
    bn_layers = watcher._get_bn_layers()

    for layer in bn_layers:
        state_dict_layer = to_device(state_dict_compressed[layer], device)
        attrgetter(layer)(model).weight = state_dict_layer['weight'].float().to(device)
        attrgetter(layer)(model).bias = state_dict_layer['bias'].float().to(device)
        attrgetter(layer)(model).running_mean.fill_(0)
        attrgetter(layer)(model).running_var.fill_(1)

    # biases of compressed layers
    biases = state_dict_compressed['biases']
    for k, v in biases.items():
        attrgetter(k)(model).data = v.to(device).float()

    # evaluate the model
    evaluate(data_loader_test, model, device)


def to_device(state_dict, device):
    return {k: v.to(device) for (k, v) in state_dict.items()}


if __name__ == '__main__':
    main()
