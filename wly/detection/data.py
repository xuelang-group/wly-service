import numpy as np
import torch


def data_loader(imgs, stride, pad_value, split_comber):
    nz, nh, nw = imgs.shape[1:]
    pz = int(np.ceil(float(nz) / stride)) * stride
    ph = int(np.ceil(float(nh) / stride)) * stride
    pw = int(np.ceil(float(nw) / stride)) * stride
    imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',
                  constant_values=pad_value)

    xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, imgs.shape[1] / stride),
                             np.linspace(-0.5, 0.5, imgs.shape[2] / stride),
                             np.linspace(-0.5, 0.5, imgs.shape[3] / stride), indexing='ij')

    coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')
    imgs, nzhw = split_comber.split(imgs)

    coord2, nzhw2 = split_comber.split(coord,
                                       side_len=int(split_comber.side_len / stride),
                                       max_stride=int(split_comber.max_stride / stride),
                                       margin=int(split_comber.margin / stride))
    assert np.all(nzhw == nzhw2)
    imgs = (imgs.astype(np.float32) - 128) / 128
    return torch.from_numpy(imgs), torch.from_numpy(coord2), np.array(nzhw)
