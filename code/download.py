# coding=utf-8
from __future__ import absolute_import, print_function

import functools
import os

from suanpan import asyncio, path, runtime
from suanpan.log import logger
from suanpan.stream import Handler as h
from suanpan.stream import Stream
from suanpan.stream.arguments import Folder, Json
from suanpan.utils import json
from wly.work_oth import downloading


class DownloadStream(Stream):
    @h.input(Json(key="inputData1", required=True))
    @h.output(Folder(key="outputData1", required=True))
    def call(self, context):
        args = context.args

        data = args.inputData1
        data_path = args.outputData1
        data_file = os.path.join(data_path, "data.json")

        series = data['series']
        windC = [int(ser['windowCenter'].split('\\')[0]) < 0 for ser in series]
        if not sum(windC):
            logger.info("Nothing to download.")
            return

        ser = series[windC.index(True)]
        dicoms_path = path.mkdirs(os.path.join(data_path, "dicoms"))

        # logger.info("Downloading: {}".format(",".join(ser['files'])))
        print("--------------------------")
        print(ser["files"])
        print("--------------------------")
        download = runtime.retry(stop_max_attempt_number=3)(functools.partial(downloading, path=dicoms_path))
        asyncio.map(download, ser['files'], thread=True, pbar=True, workers=len(ser['files']))

        data['seriesUid'] = ser['seriesUid']
        json.dump(data, data_file)

        return data_path

if __name__ == "__main__":
    DownloadStream().start()
