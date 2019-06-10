# coding=utf-8
from __future__ import absolute_import, print_function

import requests
import pandas as pd
from suanpan.log import logger
from suanpan.stream import Handler as h
from suanpan.stream import Stream
from suanpan.stream.arguments import Json
from wly.work_oth import success_ret_info

URL = "http://39.96.243.14:9191/api/gpu/submit?customStudyUid={}"


class PushStream(Stream):
    @h.input(Json(key="inputData1"))
    def call(self, context):
        args = context.args

        data = args.inputData1
        data['nodule_preb'] = pd.DataFrame(data['nodule_preb'])

        url = URL.format(data['customStudyInstanceUid'])
        response = requests.post(url, data=success_ret_info(data))
        response.raise_for_status()
        logger.info(response.text)


if __name__ == "__main__":
    PushStream().start()
