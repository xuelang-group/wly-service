import numpy as np
import scipy
import scipy.ndimage
import torch
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import (binary_dilation, binary_erosion,
                                      generate_binary_structure)
from skimage import measure
from skimage.morphology import convex_hull_image
from torch.utils.data import DataLoader
from wly.preprocessing import read_dicom
from wly.preprocessing.dataset import UDataSet


def resample(imgs, spacing, new_spacing, order=1):
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')


# def resample(imgs, spacing, new_spacing, order=2):
#     if len(imgs.shape) == 3:
#         new_shape = np.round(imgs.shape * spacing / new_spacing)
#         true_spacing = spacing * imgs.shape / new_shape
#         resize_factor = new_shape / imgs.shape
#         # resize_factor[0] = (new_shape[0] - spacing[0] + 1) / imgs.shape[0]
#         imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
#         return imgs, true_spacing
#     elif len(imgs.shape) == 4:
#         n = imgs.shape[-1]
#         newimg = []
#         for i in range(n):
#             slice = imgs[:, :, :, i]
#             newslice, true_spacing = resample(slice, spacing, new_spacing)
#             newimg.append(newslice)
#         newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
#         return newimg, true_spacing
#     else:
#         raise ValueError('wrong shape')


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def process_mask(mask, spacing, dilation_iter=10):
    ratio = float(spacing[1]) / spacing[0]
    if ratio > 0.8:
        struct = generate_binary_structure(3,1)
        dilatedMask = binary_dilation(mask,structure=struct,iterations=dilation_iter)
    else:
        struct = generate_binary_structure(3,1)
        struct[0] = False
        struct[2] = False
        xy_iter = int(dilation_iter*(1-ratio))
        dilatedMask = binary_dilation(mask,structure=struct,iterations=xy_iter)
        struct = generate_binary_structure(3,1)
        xyz_iter = dilation_iter-xy_iter
        dilatedMask = binary_dilation(dilatedMask, structure=struct, iterations=xyz_iter)
    return dilatedMask


def lumTrans(img):
    lungwin = np.array([-1200., 600.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def savenpy(im, m1, m2, spacing):
    resolution = np.array([1, 1, 1])
    # spacing[0] = 5.0
    dm1 = process_mask(m1)
    dm2 = process_mask(m2)
    dilatedMask = dm1 + dm2
    Mask = m1 + m2
    extramask = dilatedMask ^ Mask
    bone_thresh = 210
    pad_value = 170
    im[np.isnan(im)] = -2000
    sliceim = lumTrans(im)
    sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype('uint8')
    bones = sliceim * extramask > bone_thresh
    sliceim[bones] = pad_value
    sliceim, _ = resample(sliceim, spacing, resolution, order=1)
    return sliceim[np.newaxis, ...], Mask


def prepare_data(case, spacing):
    try:
        im, m1, m2 = read_dicom.read_data(case, spacing)
        # print(im.shape, spacing)
        # <class 'tuple'>: (1, 59, 512, 512)
        # <class 'tuple'>: (59, 512, 512)
        clean, Mask = savenpy(im, m1, m2, spacing)
        # print(clean.shape)
        return True, clean, spacing
    except Exception as e:
        print('prep_data error:', e)
        return False, [], []



def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):
    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw

    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    bw0 = bw0 > 0
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw0, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1: #if we have many lung regions
            if properties[0].area/properties[1].area < max_ratio: #if first two regions are reasonable
                found_flag = True
                bw1 = (label == properties[0].label) + (label == properties[1].label)
            elif properties[0].area/properties[1].area > 10: #if the first big enough then use the first
                found_flag = True
                bw1 = label == properties[0].label
            else: #erision the mask to sepeate the left and right lung
                bw0 = scipy.ndimage.binary_erosion(bw0)
                iter_count = iter_count + 1
        elif len(properties) == 1: #if we only have one region
            found_flag = True
            bw1 = label == properties[0].label
        else: #no region, return the mask or cast Exception
            return bw0
    if found_flag:
        if iter_count:
            bw1 = scipy.ndimage.binary_dilation(bw1, iterations=iter_count)

        bw1 = fill_2d_hole(bw1)
    else:
        bw1 = bw > 0

    return bw1

def prepare_data2(case, spacing, mask):
    resolution = np.array([1, 1, 1])
    sliceim = lumTrans(case)
    mask = two_lung_only(mask, spacing)
    mask = mask > 0
    zz, yy, xx = np.where(mask)
    box = np.array([[np.min(zz), np.max(zz)], [np.min(yy), np.max(yy)], [np.min(xx), np.max(xx)]])
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
    box = np.floor(box).astype('int')
    margin = 10
    extendbox = np.vstack([np.max([[0, 0, 0], box[:, 0] - margin], 0),
                           np.min([sliceim.shape * spacing, box[:, 1] + margin], axis=0).T]).T
    extendbox = extendbox.astype(int)
    dilatedMask = process_mask(mask, spacing)
    extramask = dilatedMask ^ mask

    bone_thresh = 210
    pad_value = 170

    sliceim *= dilatedMask
    sliceim += (pad_value * (1 - dilatedMask)).astype('uint8')
    bones = (sliceim * extramask) > bone_thresh
    sliceim[bones] = pad_value

    sliceim1, true_spacing = resample(sliceim, spacing, resolution, order=1)
    sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1],
               extendbox[1, 0]:extendbox[1, 1],
               extendbox[2, 0]:extendbox[2, 1]]
    return sliceim2[np.newaxis, ...], extendbox
