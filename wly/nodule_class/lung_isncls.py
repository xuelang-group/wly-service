import collections

import numpy as np
import pandas as pd
import torch
import torchvision
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import MeanShift
from torch.autograd import Variable
from torch.utils.data import DataLoader
from wly.nodule_class.dataset import TestCls_Dataset

out_size = np.array([40, 40, 20])
out_size_tr = (out_size / 2).astype(int)


def collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    if torch.is_tensor(batch[0]):
        out = None
        return torch.cat(batch, 0, out=out)
    if isinstance(batch[0], pd.DataFrame):
        return pd.concat(batch)
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]




def merge_candidates(output, distance=5.):
    voxel_dis = distance
    distances = pdist(output[:, 1:4], metric='euclidean')
    adjacency_matrix = squareform(distances)

    # Determine nodes within distance, replace by 1 (=adjacency matrix)
    adjacency_matrix = np.where(adjacency_matrix <= voxel_dis, 1, 0)
    # Determine all connected components in the graph
    n, labels = connected_components(adjacency_matrix)
    new_output = []

    # Take the mean for these connected components
    ms = MeanShift(bin_seeding=True, n_jobs=1)
    for cluster_i in range(n):
        points = output[np.where(labels == cluster_i)]
        if len(points) < 40:
            max_p = points[:, 0].max()
            max_arg = points[:, 0].argmax()
            diameter = points[max_arg, -1]
            center = np.mean(points, axis=0)
            center[0] = max_p
            center[-1] = diameter
            new_output.append(center)
            continue
        if len(points) > 400:
            points = points[np.argsort(-points[:, 0])[:400], :]
        ms.fit(points[:, 1:4])
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(ms.labels_)
        for k in range(len(labels_unique)):
            p = points[ms.labels_ == k, :]
            max_p = p[:, 0].max()
            max_arg = p[:, 0].argmax()
            diameter = p[:, -1].mean()
            diameter = p[max_arg, -1]
            center = cluster_centers[k]
            center = np.concatenate([[max_p], center, [diameter]])
            new_output.append(center)
    new_output = np.stack(new_output)
    return new_output


def pbb_to_df(labels, spacing, extendbox):
    labels = merge_candidates(labels)
    prob = labels[:, 0]
    labels = labels[:, 1:5]

    labels = labels.T
    labels[:3] = labels[:3] + np.expand_dims(extendbox[:, 0], 1)
    labels[:3] = np.round(labels[:3] / np.expand_dims(spacing, 1))
    labels[3] = labels[3] / np.expand_dims(spacing[1], 1)
    labels[:3] = labels[:3][::-1]
    labels = labels.T

    labels = np.column_stack((labels, prob))
    return pd.DataFrame(labels, columns=['coordX', 'coordY', 'coordZ', 'diameter_mm', 'probability'])


def nodule_cls(nodule_df, case, spacing, net, cuda_id):
    dataset = TestCls_Dataset(nodule_df, case, spacing, sample_num=64)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate,
        pin_memory=False)
    softmax = torch.nn.Softmax()
    probabilities_list = []
    candidates_list = []
    for i, (data, cands) in enumerate(data_loader):
        # torchvision.utils.save_image(data[:, :, :, :, 10], 'batch_%d.png' % i)
        data = Variable(data).cuda(cuda_id)
        output = net(data)
        probs = softmax(output).data[:, 1].cpu().numpy()
        probabilities_list.append(probs)
        candidates_list.append(cands)
        del output

    probabilities = np.concatenate(probabilities_list)
    candidates = pd.concat(candidates_list)
    if 'probability' in candidates.columns:
        p_1 = candidates['probability'].values
    else:
        p_1 = 1
    candidates['probability2'] = probabilities
    candidates['probability3'] = probabilities * p_1

    candidates = candidates[candidates.probability2 > 0.12]
    candidates = candidates[candidates.probability3 > 0.25]
    return candidates
