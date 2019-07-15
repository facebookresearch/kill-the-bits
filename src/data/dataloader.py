# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import utils.transforms as T
import utils.utils_mask as utils

from utils.coco_utils import get_coco, get_coco_kp
from utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups


def get_dataset(path, name, image_set, transform):
    paths = {
        "coco": (path, get_coco, 91),
        "coco_kp": (path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def load_data(path, batch_size=2, nb_workers=20, distributed=False, aspect_ratio_group_factor=0):
    """
    Loads data from ImageNet dataset.

    Args:
        - data_path: path to dataset
        - batch_size: train and test batch size
        - nb_workers: number of workers for dataloader
    """

    # data path
    print("Loading data")

    dataset, num_classes = get_dataset(path, "coco", "train", get_transform(train=True))
    dataset_test, _ = get_dataset(path, "coco", "val", get_transform(train=False))

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=nb_workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size,
        sampler=test_sampler, num_workers=nb_workers,
        collate_fn=utils.collate_fn)

    return data_loader, data_loader_test, num_classes
