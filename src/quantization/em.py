## Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import random
from collections import Counter

import torch
import numpy as np

from .solver import solve_stack
from .distance import ComputeDistances


class EM():
    """
    EM-like algorithm used to quantize the columns of M to minimize

                    ||in_activations.mm(M - M_hat)||^2

    Args:
        - n_iter: number of k-means iterations
        - n_centroids: number of centroids
        - eps: for cluster reassignment when an empty cluster is found
        - verbose: print error after each iteration

    Remarks:
        - If one cluster is empty, the most populated cluster is split into
          two clusters
        - All the relevant dimensions are specified in the code
    """

    def __init__(self, n_centroids, M, n_samples=-1, n_iter=20, eps=1e-8, verbose=True):
        # attributes
        self.n_centroids = n_centroids
        self.n_samples = n_samples
        self.n_iter = n_iter
        self.eps = eps
        self.verbose = verbose
        self.centroids = torch.Tensor()
        self.assignments = torch.Tensor()
        self.objective = []

    def initialize_centroids(self, M):
        """
        Initializes the centroids by sample random columns from M.

        Args:
            - M: weight matrix of size (in_features x out_features)
        """

        in_features, out_features = M.size()
        indices = torch.randint(low=0, high=out_features, size=(self.n_centroids,)).long()
        self.centroids = M[:, indices].t()  # (n_centroids x in_features)

    def step(self, in_activations, in_activations_eval, M, i):
        """
        There are two standard steps for each iteration: expectation (E) and
        minimization (M). The E-step (assignment) is performed with an exhaustive
        search and the M-step (centroid computation) is performed with a solver.

        Args:
            - in_activations: input activations of size (n_samples x in_features)
            - M: weight matrix of size (in_features x out_features)

        Remarks:
            - The E-step heavily uses PyTorch broadcasting to speed up computations
              and reduce the memory overhead
            - The M-step uses a solver with a pre-computed pseudo-inverse so its
              complexity is only one matrix multiplication
            - With the size constraints, we have out_activations = in_activations.mm(M)
            - Evaluation on a fixed batch of activations
        """

        # network for parallelization of computations
        self.compute_distances_parallel = ComputeDistances(M, self.centroids)

        # pre-compute A_pinv to factorize computations, on CPU to avoid CUDA oom error
        A_pinv = torch.pinverse(in_activations)

        # assignments (E-step)
        distances = self.compute_distances(in_activations)  # (n_centroids x out_features)
        self.assignments = torch.argmin(distances, dim=0)   # (out_features)

        # empty clusters
        counts = Counter(map(lambda x: x.item(), self.assignments))
        empty_clusters = set(range(self.n_centroids)) - set(counts.keys())
        n_empty_clusters = len(empty_clusters)

        while len(empty_clusters) > 0:
            # given an empty cluster, find most populated cluster and split it into two
            k = random.choice(list(empty_clusters))
            m = counts.most_common(1)[0][0]
            e = torch.randn_like(self.centroids[m]) * self.eps
            self.centroids[k] = self.centroids[m].clone()
            self.centroids[k] += e
            self.centroids[m] -= e
            print(k, len(empty_clusters))

            # recompute assignments
            distances = self.compute_distances(in_activations)  # (n_centroids x out_features)
            self.assignments = torch.argmin(distances, dim=0)   # (out_features)

            # check for empty clusters
            counts = Counter(map(lambda x: x.item(), self.assignments))
            empty_clusters = set(range(self.n_centroids)) - set(counts.keys())

        # centroids (M-step)
        for k in range(self.n_centroids):
            M_k = M[:, self.assignments == k]  # (in_features x size_of_cluster_k)
            B = in_activations.mm(M_k)
            self.centroids[k] = solve_stack(A=in_activations, B=B, A_pinv=A_pinv)  # (in_features)

        # book-keeping
        n_samples_eval = 128
        in_activations_eval = in_activations_eval[:n_samples_eval]
        normalize = np.sqrt(n_samples_eval * len(self.assignments))  # np.sqrt(out_activations.numel())
        obj = (in_activations_eval.mm(self.centroids[self.assignments].t() - M)).norm(p=2).div(normalize).item()  # (n_samples x in_features).mm((out_features x in_features).t()) -> (n_samples x out_features) -> 1
        self.objective.append(obj)
        if self.verbose: print("Iteration: {},\t objective: {:.6f},\t resolved empty clusters: {}".format(i, obj, n_empty_clusters))

    def compute_distances(self, in_activations):
        """
        For every centroid m and every input activation in_activation, computes

                          ||in_activations.mm(M - m[None, :])||_2

        Args:
            - in_activations: input activations of size (n_samples x in_features)
            - M: weight matrix of size (in_features x out_features)
            - centroids: centroids of size (n_centroids x in_features)

        Remarks:
            - We rely on PyTorch's broadcasting to speed up computations
              and reduce the memory overhead
            - Without chunking, the sizes in the broadcasting are modified as:
              (n_centroids x n_samples x out_features) -> (n_centroids x out_features)
            - The broadcasting computation is automatically chunked so that
              the tensors fit into the memory of the GPU
        """

        self.compute_distances_parallel.update_centroids(self.centroids)
        return self.compute_distances_parallel(in_activations)

    def assign(self, in_activations, M):
        """
        Assigns each column of M to its closest centroid, thus essentially
        performing the E-step in train().

        Args:
            - in_activations: input activations of size (n_samples x in_features)
            - M: weight matrix of size (in_features x out_features)

        Remarks:
            - The function must be called after train() or after loading
              centroids using self.load(), otherwise it will return empty tensors
            - The assignments may differ from self.assignments when this function
              is called with distinct parameters in_activations and M
        """

        # network for parallelization of computations
        self.compute_distances_parallel = ComputeDistances(M, self.centroids)

        distances = self.compute_distances(in_activations)  # (n_centroids x out_features)
        assignments = torch.argmin(distances, dim=0)        # (out_features)

        return assignments

    def save(self, path, layer):
        """
        Saves centroids and assignments.

        Args:
            - path: folder used to save centroids and assignments
        """

        torch.save(self.centroids, os.path.join(path, '{}_centroids.pth'.format(layer)))
        torch.save(self.assignments, os.path.join(path, '{}_assignments.pth'.format(layer)))
        torch.save(self.objective, os.path.join(path, '{}_objective.pth'.format(layer)))

    def load(self, path, layer):
        """
        Loads centroids and assignments from a given path

        Args:
            - path: folder use to load centroids and assignments
        """

        self.centroids = torch.load(os.path.join(path, '{}_centroids.pth'.format(layer)))
        self.assignments = torch.load(os.path.join(path, '{}_assignments.pth'.format(layer)))
        self.objective = torch.load(os.path.join(path, '{}_objective.pth'.format(layer)))
