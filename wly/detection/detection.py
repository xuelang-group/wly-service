import time

import numpy as np
import torch
from torch.autograd import Variable
from wly.detection.data import data_loader


def prediction(net, get_pbb, img, split_comber, result_dict, cuda_id):

    # stride = 4
    # pad_value = 170
    # imgs, coord, nzhw = data_loader(img, stride, pad_value, split_comber)
    imgs = result_dict['dete_imgs']
    coord = result_dict['dete_coord']
    nzhw = result_dict['dete_nzhw']

    splitlist = range(0, len(imgs) + 1, 1)
    if splitlist[-1] != len(imgs):
        splitlist.append(len(imgs))
    outputlist = []

    for i in range(len(splitlist) - 1):
        # img: torch.Size([1, 1, 208, 208, 208])
        input = Variable(imgs[splitlist[i]:splitlist[i + 1]]).cuda(cuda_id)
        inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]]).cuda(cuda_id)
        output = net(input, inputcoord)
        # torch.cuda.empty_cache()
        # print('out_put:',output.data.cpu().numpy().shape)
        outputlist.append(output.data.cpu().numpy())
        del output

    output = np.concatenate(outputlist, 0)

    output = split_comber.combine(output, nzhw=nzhw)
    thresh = -3
    pbb, mask = get_pbb(output, thresh, ismask=True)
    torch.cuda.empty_cache()
    return pbb
