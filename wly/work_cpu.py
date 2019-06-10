import gc
import threading
import time
from multiprocessing import Pool

from func_timeout import FunctionTimedOut
from torch.backends import cudnn
from wly.detection import split_combine
from wly.detection.data import data_loader
from wly.detection.res18 import get_pbb
from wly.preprocessing import read_dicom
from wly.preprocessing.lung_segment import LungSegmentUnet
from wly.preprocessing.prepare import prepare_data2
from wly.utils.ret_utils import error_info


class CpuThread(threading.Thread):
    def __init__(self, que_pre, que_det):
        threading.Thread.__init__(self)
        self.que_pre = que_pre
        self.que_det = que_det
        self.lung_segm = LungSegmentUnet('./model/lung_segment.ckpt')
        cudnn.benchmark = True

        self.get_pbb = get_pbb()
        max_stride = 16
        margin = 32
        stride = 4
        sidelen = 144
        pad_value = 170
        self.split_comber = split_combine.SplitComb(sidelen, max_stride, stride, margin, pad_value)
        self.pool = Pool(processes=1)

    def run(self):
        while True:
            if self.que_pre.qsize() > 0:
                result_dict = self.que_pre.get()
                try:
                    t_s = time.time()
                    # res = self.pool.apply_async(cpu_preprocess_1, (result_dict,))
                    # case, spacing, instances = res.get()
                    case, spacing, instances = read_dicom.load_dicom2(result_dict['data_path'])
                    print('load us :',time.time()-t_s)
                    # assert 40 < case.shape[0] < 80
                    prep_mask = self.lung_segm.cut(case, 30)

                    res = self.pool.apply_async(cpu_preprocess_2, (case, spacing, prep_mask,self.split_comber))
                    prep_data, extendbox, imgs, coord, nzhw = res.get()
                    # stride = 4
                    # pad_value = 170
                    #
                    # prep_data, extendbox = prepare_data2(case, spacing, prep_mask)
                    result_dict['prep_case'] = case
                    result_dict['prep_spac'] = spacing
                    result_dict['prep_inst'] = instances
                    result_dict['prep_data'] = prep_data
                    result_dict['prep_mask'] = prep_mask
                    result_dict['prep_ebox'] = extendbox
                    # imgs, coord, nzhw = data_loader(result_dict['prep_data'], stride, pad_value, self.split_comber)
                    result_dict['dete_imgs'] = imgs
                    result_dict['dete_coord'] = coord
                    result_dict['dete_nzhw'] = nzhw
                    print(time.ctime() + 'cpu process task us time: {}.{}'.format(time.time() - t_s,
                                                                                  result_dict['data_path']))
                    self.que_det.put(result_dict)
                    gc.collect()
                except FunctionTimedOut:
                    print(time.ctime() + 'FUN TIMEOUT ')
                except Exception as e:
                    print (e)
                    error_info(100, result_dict)
            else:
                time.sleep(1)



def cpu_preprocess_1(result_dict):
    case, spacing, instances = read_dicom.load_dicom2(result_dict['data_path'])
    return case, spacing, instances


def cpu_preprocess_2(case, spacing, prep_mask, split_comber):
    prep_data, extendbox = prepare_data2(case, spacing, prep_mask)
    stride = 4
    pad_value = 170
    imgs, coord, nzhw = data_loader(prep_data, stride, pad_value, split_comber)

    return prep_data, extendbox, imgs, coord, nzhw
