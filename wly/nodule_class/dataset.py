import numpy as np
import pandas as pd
import torch
import wly.nodule_class.config as cfg
from torch.utils.data import Dataset


def lumTrans(img):
    lungwin = np.array([-1000., 400.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


class TestCls_Dataset(Dataset):
    def __init__(self, candidates, lung, spacing, sample_num=64):
        """ candidates: pandas DataFrame
            sample_num: samples of each lung.
        """
        self.sample_num = sample_num
        self.candidates = candidates
        self.spacing = spacing
        # normalize the CT
        self.zero_center = cfg.ZERO_CENTER
        self.mean_pixel = cfg.MEAN_PIXEL
        lung = lung.astype(np.float32)
        lung = self.normalize(lung)
        if self.zero_center:
            lung -= self.mean_pixel

        self.lung = np.transpose(lung)

        self.batch_list = []
        count = 0
        for cand in self.candidates.values:
            if count % sample_num == 0:
                self.batch_list.append([])
            self.batch_list[-1].append(cand)
            count += 1

        # load config
        self.cube_size = np.array(cfg.CUBE_SIZE)
        self.input_size = np.array(cfg.INPUT_SIZE)
        self.phase = 'test'

    def __getitem__(self, idx):
        batch = self.batch_list[idx]
        lung = self.lung
        # origin = self.origin
        spacing = self.spacing
        # load data
        patches = []
        for cand in batch:
            coords = cand[0:3]
            patch = self.get_single_patch(lung, coords)
            patches.append(patch)
        patches = torch.stack([torch.from_numpy(p[np.newaxis, ...]) for p in patches], 0)
        return patches, pd.DataFrame(batch, columns=self.candidates.columns)  # tensor, DataFrame

    def get_single_patch(self, lung, coords):
        # coords = (np.floor(world_2_voxel(coords[:3],origin,spacing))).astype(np.int64)
        # print(coords, 'lung_shape:', lung.shape)
        output = np.zeros(self.input_size, dtype=np.float32)
        offset = self.input_size // 2
        start = []
        end = []

        for i in range(3):
            start.append(max(0, coords[i] - offset[i]))
            end.append(min(lung.shape[i], coords[i] + offset[i]))
        start = np.array(start).astype(int)
        end = np.array(end).astype(int)
        center = self.input_size // 2
        cube_s = center - (coords - start)
        cube_e = center + (end - coords)
        cube_s = cube_s.astype(int)
        cube_e = cube_e.astype(int)
        # print(start,end)
        # print(cube_s,cube_e)
        output[cube_s[0]:cube_e[0], cube_s[1]:cube_e[1], cube_s[2]:cube_e[2]] = \
            lung[start[0]: end[0], start[1]: end[1], start[2]: end[2]]
        assert output.shape == (self.input_size[0], self.input_size[1], self.input_size[2])
        return output

    def __len__(self):
        return len(self.batch_list)

    def normalize(self, image):
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0
        image = np.clip(image, MIN_BOUND, MAX_BOUND, image)
        image -= MIN_BOUND
        image /= (MAX_BOUND - MIN_BOUND)
        return image
