import numpy as np
import torch
from func_timeout import func_set_timeout
from torch.autograd import Variable
from torch.nn import DataParallel
from wly.preprocessing.segment.unet import UNet


def lumTrans(img):
    img = img.astype(np.float)
    lungwin = np.array([-1200., 600.])
    img = np.clip(img, lungwin[0], lungwin[1], img)
    img -= lungwin[0]
    img *= (255. / (lungwin[1] - lungwin[0]))
    return img.astype(np.float32)


class LungSegmentUnet(object):
    def __init__(self, model):
        self.net = UNet(color_dim=3, num_classes=3)
        checkpoint = torch.load(model)
        self.net.load_state_dict(checkpoint['state_dir'])
        self.net = DataParallel(self.net).cuda()
        self.net.eval()

    @func_set_timeout(20)
    def cut(self, data, b_s=64):
        """ input:
                data: D * H * W ndarray
            output:
                [0,1,2] ^ (D * H * W) uint8 mask
        """
        stride = 16
        data = lumTrans(data)
        slices, h, w = data.shape

        pad_h = pad_w = 0
        if h % stride != 0:
            pad_h = stride - h % stride
            pad = np.zeros((slices, pad_h, w), dtype=np.uint8)
            data = np.concatenate([pad, data], axis=1)
        if w % stride != 0:
            pad_w = stride - w % stride
            pad = np.zeros((slices, h + pad_h, pad_w), dtype=np.uint8)
            data = np.concatenate([pad, data], axis=2)
        p_h, p_w = h + pad_h, w + pad_w

        data = np.expand_dims(data, 1)

        x, y = np.meshgrid(np.linspace(-1, 1, p_h), np.linspace(-1, 1, p_w), indexing='ij')
        x, y = x.astype(np.float32)[np.newaxis, :], y.astype(np.float32)[np.newaxis, :]
        # concat
        data = np.stack([np.concatenate([d, x, y], 0) for d in data])  # D*3*H*W

        splitlist = list(range(0, len(data) + 1, b_s))
        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []

        tensor = Variable(torch.from_numpy(data), volatile=True)

        for i in range(len(splitlist) - 1):
            input = tensor[splitlist[i]:splitlist[i + 1]].cuda()
            batch_size = len(input)

            output = self.net(input)
            output = output.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 3)
            _, output = output.data.max(dim=1)
            output[output > 0] += 2
            output = output.view(batch_size, p_h, p_w)
            outputlist.append(output.cpu().numpy())
        output = np.concatenate(outputlist, axis=0).astype(np.uint8)
        if pad_w > 0:
            output = output[:, :, pad_w:]
        if pad_h > 0:
            output = output[:, pad_h:, :]
        return output
