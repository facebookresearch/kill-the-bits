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

import models as resnet_models
import torchvision.models.detection as detection_models
from data import load_data
from utils.training import evaluate
from utils.watcher import ActivationWatcher as ActivationWatcherResNet
from utils.utils import weight_from_centroids


parser = argparse.ArgumentParser(description='Inference for quantized networks')
parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet50', 'resnet50_semisup'],
                    help='Model to use for inference')
parser.add_argument('--state-dict-compressed', default='', type=str,
                    help='Path to the compressed state dict of the model')
parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'],
                    help='For inference on CPU or on GPU')
parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                    help='Path to ImageNet dataset')
parser.add_argument('--batch-size', default=128, type=int,
                    help='Batch size for fiuetuning steps')
parser.add_argument('--n-workers', default=20, type=int,
                    help='Number of workers for data loading')


def main():
    global args
    args = parser.parse_args()
    device = args.device
    state_dict_compressed = torch.load(args.state_dict_compressed)

    # instantiating model
    model = 'resnet50' if args.model == 'resnet50_semisup' else args.model
    model = resnet_models.__dict__[model](pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    _, test_loader = load_data(data_path=args.data_path, batch_size=args.batch_size, nb_workers=args.n_workers)
    watcher = ActivationWatcherResNet(model)

    # conv1 layer (non-compressed)
    layer = 'conv1'
    state_dict_layer = to_device(state_dict_compressed[layer], device)
    attrgetter(layer)(model).load_state_dict(state_dict_layer)
    attrgetter(layer)(model).float()

    # compressed layers
    compressed_layers = watcher.layers[1:]

    # 2 more layers non-compressed for semi-supervised ResNet50
    if args.model == 'resnet50_semisup':
        non_compressed_layers = ['layer1.0.conv3', 'layer1.0.downsample.0']
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
        attrgetter(layer)(model).weight.data = state_dict_layer['weight'].float().to(device)
        attrgetter(layer)(model).bias.data = state_dict_layer['bias'].float().to(device)

    # classifier bias
    layer = 'fc'
    state_dict_layer = to_device(state_dict_compressed['fc_bias'], device)
    attrgetter(layer + '.bias')(model).data = state_dict_layer['bias']

    # evaluate the model
    top_1 = evaluate(test_loader, model, criterion, device=device).item()
    print('Top-1 accuracy of quantized model: {:.2f}'.format(top_1))


def to_device(state_dict, device):
    return {k: v.to(device) for (k, v) in state_dict.items()}


if __name__ == '__main__':
    main()
