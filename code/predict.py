# coding=utf-8
from __future__ import absolute_import, print_function

import os

import cv2
import numpy as np
import torch
from func_timeout import FunctionTimedOut
from suanpan import asyncio
from suanpan.log import logger
from suanpan.stream import Handler as h
from suanpan.stream import Stream
from suanpan.stream.arguments import Bool, Folder, Int, Json, Npy
from suanpan.utils import convert, image, json, npy, pickle
from torch.backends import cudnn
from wly.work_gpu import LungDetection, LungIsncls, lobe_locate_gmm

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def rectangle(img, box, color=RED, width=1, *arg, **kwargs):
    cy, cx, aa = box
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # pylint: disable=no-member
    return cv2.rectangle(  # pylint: disable=no-member
        img, (cx - aa, cy - aa), (cx + aa, cy + aa), color, width, *arg, **kwargs
    )


def pickWithPbb(img, pbb, maskFunc=rectangle, *arg, **kwargs):
    image3D = convert.to3D(img)

    def _mask(box):
        box = box.astype("int")[1:]
        zindex, box = box[0], box[1:]
        return addMask(image3D[zindex], box, maskFunc=rectangle, *arg, **kwargs)

    return (_mask(box) for box in pbb)


def addMask(image, box, maskFunc=rectangle, *arg, **kwargs):
    return maskFunc(image, box, *arg, **kwargs)


def fixed3(value):
    return "{0:0=3d}".format(value)


class PredictStream(Stream):

    ARGUMENTS = [Int(key="param1", alias="batch", default=1)]

    def afterInit(self):
        self.lung_dete = LungDetection("model/det.ckpt")
        self.lung_isnc = LungIsncls("model/isn.ckpt")
        cudnn.benchmark = True

        self.left_gmm = pickle.load(open("model/left_gmm.pkl", "rb"), encoding="latin1")
        self.right_gmm = pickle.load(
            open("model/right_gmm.pkl", "rb"), encoding="latin1"
        )

    @h.input(Folder(key="inputData1", required=True))
    @h.output(Json(key="outputData1"))
    @h.output(Npy(key="outputData2"))
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
        case, spacing, prep_data, prep_mask, extendbox, imgs, coord, nzhw = asyncio.map(
            npy.load, npys, thread=True, pbar=True, workers=len(npys)
        )
        imgs = torch.from_numpy(imgs)
        coord = torch.from_numpy(coord)

        try:
            nodule_df, pbb = self.lung_dete.prediction(
                imgs, coord, nzhw, spacing, extendbox, batch=self.args.batch
            )
            preb = self.lung_isnc.nodule_cls(nodule_df, case, spacing)
            preb = self.lung_lobe(preb, case, prep_mask, spacing)
            data["nodule_preb"] = preb.to_dict()

            logger.info("Sending predicted images...")
            images = np.array([img for img in pickWithPbb(prep_data, pbb)])

            return data, convert.flatAsImage(images)
        except FunctionTimedOut:
            raise Exception("Predict timeout")

    def lung_lobe(self, nodule_df, case, mask, spacing):
        nodule_df_values = nodule_df[["coordX", "coordY", "coordZ"]].values
        lungs = []
        lobes = []
        longs = []
        shorts = []

        lobel_info = []
        for nodule in nodule_df_values:
            lung, lobe = lobe_locate_gmm(
                nodule, case, mask, spacing, self.left_gmm, self.right_gmm
            )
            lungs.append(lung)
            lobes.append(lobe)
            lobel_info.append(lung + "肺" + (lobe + "叶" if not lobe == "" else ""))

        nodule_df["lung"] = lungs
        nodule_df["lobe"] = lobes
        nodule_df["lobel_info"] = lobel_info
        return nodule_df


if __name__ == "__main__":
    PredictStream().start()
