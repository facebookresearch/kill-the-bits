# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from operator import attrgetter

from .training import evaluate


class ActivationWatcher:
    """
    Monitors and stores *input* activations in all the layers of the network.

    Args:
        - model: the model to monitor, should be `nn.module`
        - n_activations: number of activations to store
        - layer: if None, monitors all layers except BN, otherwise single
          layers to monitor

    Remarks:
        - Do NOT use layers with inplace operations, otherwise
          the activations will not be monitored correctly
        - Memory to store activations is pre-allocated for efficiency
    """

    def __init__(self, model, layer=''):
        self.model = model
        # layers to monitor
        all_layers = self._get_layers()
        if len(layer) == 0:
            self.layers = all_layers
        else:
            assert layer in all_layers
            self.layers = [layer]
        # initialization
        self.modules_to_layers = {attrgetter(layer)(self.model): layer for layer in self.layers}
        self._register_hooks()
        self._watch = False

    def _get_layers(self):
        # get proper layer names
        keys = self.model.state_dict().keys()
        layers = [k[:k.rfind(".")] for k in keys if 'bias' not in k]
        # remove BN layers
        layers = [layer for layer in layers if not isinstance(attrgetter(layer)(self.model), nn.BatchNorm2d)]

        return layers

    def _get_bn_layers(self):
        # get proper layer names
        keys = self.model.state_dict().keys()
        layers = [k[:k.rfind(".")] for k in keys if 'weight' in k]
        # only keep BN layers
        layers = [layer for layer in layers if isinstance(attrgetter(layer)(self.model), nn.BatchNorm2d)]

        return layers

    def _get_bias_layers(self):
        # get proper layer names
        keys = self.model.state_dict().keys()
        layers = [k[:k.rfind(".")] for k in keys if 'bias' in k]
        # only keep BN layers
        layers = [layer for layer in layers if not isinstance(attrgetter(layer)(self.model), nn.BatchNorm2d)]

        return layers

    def _register_hooks(self):
        # define hook to save output after each layer
        def fwd_hook(module, input, output):
            layer = self.modules_to_layers[module]
            if self._watch:
                # retrieve activations
                activations = input[0].data.cpu()
                # store activations
                self.activations[layer].append(activations)
        # register hooks
        self.handles = []
        for layer in self.layers:
            handle = attrgetter(layer)(self.model).register_forward_hook(fwd_hook)
            self.handles.append(handle)

    def watch(self, loader, criterion, n_iter):
        # watch
        self._watch = True
        # initialize activations storage
        self.activations = {layer: [] for layer in self.layers}
        # gather activations
        evaluate(loader, self.model, criterion, n_iter=n_iter)
        # unwatch
        self._watch = False
        # treat activations
        self.activations = {k: torch.cat(v, dim=0) for (k, v) in self.activations.items()}
        # remove hooks from model
        for handle in self.handles:
            handle.remove()
        # return activations
        return self.activations

    def save(self, path):
        torch.save(self.activations, path)
