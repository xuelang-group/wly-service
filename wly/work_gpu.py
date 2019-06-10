import gc
import pickle
import threading
import time

import numpy as np
import torch
from func_timeout import FunctionTimedOut
from skimage import filters, measure, morphology, segmentation
from torch.backends import cudnn
from torch.nn import DataParallel

from wly.detection import res18, split_combine
from wly.detection.detection import prediction
from wly.detection.lung_detection import LungDetection
from wly.nodule_class.conv6 import Net
from wly.nodule_class.isnodule import LungIsncls
from wly.nodule_class.lung_isncls import nodule_cls, pbb_to_df
from wly.preprocessing.location import lobe_locate_gmm
from wly.utils.det_utils import find_Dfs, nms


class GpuThread(threading.Thread):
    def __init__(self, que_det, que_ret):
        threading.Thread.__init__(self)
        self.que_det = que_det
        self.que_ret = que_ret
        self.lung_dete = LungDetection('./model/det.ckpt')

        # is nodule cls
        self.lung_isnc = LungIsncls('./model/isn.ckpt')

        l_u = pickle._Unpickler(open('./model/left_gmm.pkl', 'rb'))
        l_u.encoding = 'latin1'
        self.left_gmm = l_u.load()

        r_u = pickle._Unpickler(open('./model/right_gmm.pkl', 'rb'))
        r_u.encoding = 'latin1'
        self.right_gmm = r_u.load()
        cudnn.benchmark = True

    def run(self):
        while True:
            try:
                if self.que_det.qsize() > 0:
                    t_s = time.time()
                    print(' GPU WORKER DOING  ')
                    result_dict = self.que_det.get(timeout=5)
                    t_s = time.time()
                    nodule_df = self.lung_dete.prediction(result_dict['dete_imgs'], result_dict['dete_coord'],
                                                          result_dict['dete_nzhw'], result_dict['prep_spac'],
                                                          result_dict['prep_ebox'])
                    preb = self.lung_isnc.nodule_cls(nodule_df, result_dict['prep_case'], result_dict['prep_spac'])
                    preb = self.lung_lobe(preb, result_dict['prep_case'], result_dict['prep_mask'],
                                          result_dict['prep_spac'])
                    result_dict['nodule_preb'] = preb
                    self.que_ret.put(result_dict, timeout=5)
                    gc.collect()
                    print(time.ctime() + 'GPU DOING US TIME:', time.time() - t_s)
                else:
                    time.sleep(1)
            except FunctionTimedOut:
                print(time.ctime() + 'FUN TIMEOUT ')
            except Exception as e:
                print(time.ctime() + 'GPU ERROR : {} '.format(e))


    def lung_lobe(self, nodule_df, case, mask, spacing):
        s_time = time.time()
        nodule_df_values = nodule_df[['coordX', 'coordY', 'coordZ']].values
        lungs = []
        lobes = []
        longs = []
        shorts = []

        lobel_info = []
        for nodule in nodule_df_values:
            lung, lobe = lobe_locate_gmm(nodule, case, mask, spacing, self.left_gmm, self.right_gmm)
            lungs.append(lung)
            lobes.append(lobe)
            lobel_info.append(lung + '肺' + (lobe + '叶' if not lobe == '' else ''))
            # Long_short_axis
            # long, short = long_short_axis(nodule, clean)
            # longs.append(long)
            # shorts.append(short)
        nodule_df['lung'] = lungs
        nodule_df['lobe'] = lobes
        # nodule_df['short'] = shorts
        # nodule_df['long'] = longs
        nodule_df['lobel_info'] = lobel_info
        return nodule_df
