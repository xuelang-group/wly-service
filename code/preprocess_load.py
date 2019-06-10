# coding=utf-8
from __future__ import absolute_import, print_function

import os

from func_timeout import FunctionTimedOut
from suanpan import asyncio, path
from suanpan.stream import Handler as h
from suanpan.stream import Stream
from suanpan.stream.arguments import Folder
from suanpan.utils import json, npy
from wly.work_cpu import read_dicom


class PreprocessStream(Stream):
    @h.input(Folder(key="inputData1", required=True))
    @h.output(Folder(key="outputData1", required=True))
    def call(self, context):
        args = context.args

        input_path = args.inputData1
        dicoms_path = os.path.join(input_path, "dicoms")
        data = json.load(os.path.join(input_path, "data.json"))

        try:
            case, spacing, instances = read_dicom.load_dicom2(dicoms_path)
        except FunctionTimedOut:
            raise Exception("Preprocess timeout")

        output_path = args.outputData1
        output_npys_path = path.mkdirs(os.path.join(output_path, "npys"))

        data["prep_inst"] = instances
        json.dump(data, os.path.join(output_path, "data.json"))
        npys = [
            (case, os.path.join(output_npys_path, "case.npy")),
            (spacing, os.path.join(output_npys_path, "spacing.npy")),
        ]
        asyncio.starmap(npy.dump, npys, thread=True, pbar=True, workers=len(npys))

        return output_path


if __name__ == "__main__":
    PreprocessStream().start()
