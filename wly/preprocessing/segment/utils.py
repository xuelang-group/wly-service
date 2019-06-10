import os
import sys

import numpy as np
import SimpleITK as sitk


def load_itk(filename):
    """return image, origin, spacing,
       image in z, y, x order;
       origin and spacing in x, y, z order.
    """
    itkimage = sitk.ReadImage(filename)
    image = sitk.GetArrayFromImage(itkimage)
    origin = np.array(itkimage.GetOrigin())
    spacing = np.array(itkimage.GetSpacing())
    return image, origin, spacing

def save_itk(image, origin, spacing, filename):
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)

def world_2_voxel(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    voxel_coord = voxel_coord.astype(np.int32)
    return voxel_coord

def voxel_2_world(voxel_coord, origin, spacing):
    stretched_voxel_coord = voxel_coord * spacing
    world_coord = stretched_voxel_coord + origin
    return world_coord

def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)


def weight_by_class_balance(truth, classes=None):
    """
    Determines a loss weight map given the truth by balancing the classes from the classes argument.
    The classes argument can be used to only include certain classes (you may for instance want to exclude the background).
    """

    if classes is None:
        # Include all classes
        classes = np.unique(truth)

    weight_map = np.zeros_like(truth, dtype=np.float32)
    total_amount = np.product(truth.shape)

    for c in classes:
        class_mask = np.where(truth==c,1,0)
        class_weight = 1/((np.sum(class_mask)+1e-8)/total_amount)

        weight_map += (class_mask*class_weight)#/total_amount

    return weight_map

def iou(box0, box1):
    '''
        overlap of two [z, y, x, d] boxes
    '''

    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union
