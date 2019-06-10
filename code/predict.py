# coding=utf-8
from __future__ import absolute_import, print_function

import os

import cv2
import torch
from func_timeout import FunctionTimedOut
from suanpan import asyncio
from suanpan.stream import Handler as h
from suanpan.stream import Stream
from suanpan.stream.arguments import Bool, Folder, Int, Json, Npy
from suanpan.utils import convert, image, json, npy, pickle
from torch.backends import cudnn
from wly.work_gpu import LungDetection, LungIsncls, lobe_locate_gmm


class PredictStream(Stream):

    ARGUMENTS = [Int(key="param1", alias="batch", default=1)]

    def afterInit(self):
        self.lung_dete = LungDetection('model/det.ckpt')
        self.lung_isnc = LungIsncls('model/isn.ckpt')
        cudnn.benchmark = True

        self.left_gmm = pickle.load(open('model/left_gmm.pkl', 'rb'), encoding='latin1')
        self.right_gmm = pickle.load(open('model/right_gmm.pkl', 'rb'), encoding='latin1')

    @h.input(Folder(key="inputData1", required=True))
    @h.output(Json(key="outputData1"))
    def call(self, context):
        args = context.args

        input_path = args.inputData1
        npys_path = os.path.join(input_path, "npys")
        data = json.load(os.path.join(input_path, "data.json"))

        npys = [
            os.path.join(npys_path, "case.npy"),
            os.path.join(npys_path, "spacing.npy"),
            os.path.join(npys_path, "prep_data.npy"),
            os.path.join(npys_path, "prep_mask.npy"),
            os.path.join(npys_path, "extendbox.npy"),
            os.path.join(npys_path, "imgs.npy"),
            os.path.join(npys_path, "coord.npy"),
            os.path.join(npys_path, "nzhw.npy"),
        ]
        case, spacing, prep_data, prep_mask, extendbox, imgs, coord, nzhw = \
            asyncio.map(npy.load, npys, thread=True, pbar=True, workers=len(npys))
        imgs = torch.from_numpy(imgs)
        coord = torch.from_numpy(coord)

        try:
            nodule_df = self.lung_dete.prediction(imgs, coord, nzhw, spacing, extendbox, batch=self.args.batch)
            preb = self.lung_isnc.nodule_cls(nodule_df, case, spacing)
            preb = self.lung_lobe(preb, case, prep_mask, spacing)
            data['nodule_preb'] = preb.to_dict()
        except FunctionTimedOut:
            raise Exception("Predict timeout")

        return data

    def lung_lobe(self, nodule_df, case, mask, spacing):
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

        nodule_df['lung'] = lungs
        nodule_df['lobe'] = lobes
        nodule_df['lobel_info'] = lobel_info
        return nodule_df


if __name__ == "__main__":
    PredictStream().start()
