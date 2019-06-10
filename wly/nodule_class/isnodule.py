import collections

import numpy as np
import pandas as pd
import torch
from func_timeout import func_set_timeout
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from wly.nodule_class.conv6 import Net
from wly.nodule_class.dataset import TestCls_Dataset


def collate(batch):
    if torch.is_tensor(batch[0]):
        out = None
        return torch.cat(batch, 0, out=out)
    if isinstance(batch[0], pd.DataFrame):
        return pd.concat(batch)
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]


class LungIsncls(object):
    def __init__(self, model_path):
        isn_net = Net()
        isn_net.load_state_dict(torch.load(model_path))
        self.isn_net = DataParallel(isn_net).cuda()
        self.isn_net.eval()
        del isn_net

    @func_set_timeout(20)
    def nodule_cls(self, nodule_df, case, spacing):
        dataset = TestCls_Dataset(nodule_df, case, spacing, sample_num=32)
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
            data = Variable(data,  volatile=True).cuda()
            output = self.isn_net(data)
            probs = softmax(output).data[:, 1].cpu().numpy()
            probabilities_list.append(probs)
            candidates_list.append(cands)
            del data, output

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
