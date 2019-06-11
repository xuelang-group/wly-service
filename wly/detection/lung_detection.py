import numpy as np
import torch
from func_timeout import func_set_timeout
from torch.autograd import Variable
from torch.nn import DataParallel
from wly.detection import res18, split_combine
from wly.nodule_class.lung_isncls import pbb_to_df
from wly.utils.det_utils import nms


class LungDetection(object):
    def __init__(self, model_path):
        max_stride = 16
        margin = 32
        stride = 4
        sidelen = 144
        pad_value = 170
        self.split_comber = split_combine.SplitComb(sidelen, max_stride, stride, margin, pad_value)

        # detection net
        config1, nod_net, loss, get_pbb = res18.get_model()
        checkpoint = torch.load(model_path)
        nod_net.load_state_dict(checkpoint)
        self.get_pbb = get_pbb
        self.nod_net = DataParallel(nod_net).cuda()
        self.nod_net.eval()
        del nod_net

    @func_set_timeout(40)
    def prediction(self, imgs, coord, nzhw, spacing, endbox, batch=1):
        splitlist = list(range(0, len(imgs), batch))
        if splitlist[-1] != len(imgs):
            splitlist.append(len(imgs))

        def _run(start, end):
            with torch.no_grad():
                input = Variable(imgs[start:end]).cuda()
                inputcoord = Variable(coord[start:end]).cuda()
                output = self.nod_net(input, inputcoord)
                result =  output.data.cpu().numpy()
                del input, inputcoord, output
                return result

        outputlist = [_run(start, end) for start, end in zip(splitlist[:-1], splitlist[1:])]
        output = np.concatenate(outputlist, 0)

        output = self.split_comber.combine(output, nzhw=nzhw)
        thresh = -3
        pbb, _ = self.get_pbb(output, thresh, ismask=True)
        torch.cuda.empty_cache()
        pbb = nms(pbb, 0.05)
        nodule_df = pbb_to_df(pbb, spacing, endbox)
        nodule_df = nodule_df[nodule_df.probability > 0.25]
        return nodule_df, pbb
