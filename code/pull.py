# coding=utf-8
from __future__ import absolute_import, print_function

import requests
from suanpan.log import logger
from suanpan.stream import Handler as h
from suanpan.stream import Trigger
from suanpan.stream.arguments import Float, Json

URL = "http://39.96.243.14:9191/api/gpu/next?modality=CT&st=5"


class TriggerDemo(Trigger):
    ARGUMENTS = [Float(key="param1", alias="interval", default=20)]

    def afterInit(self):
        self.interval = self.args.interval

    @h.output(Json(key="outputData1"))
    def trigger(self, context):  # pylint: disable=unused-argument
        response = requests.get(URL, timeout=3, params={"env": "test"})
        response.raise_for_status()
        result = response.json()

        if not result:
            raise Exception("Invalid Result: {}".format(result))

        errcode = result.get("errCode")
        if errcode != 0:
            raise Exception("Response Error: {}".format(errcode))

        val = result.get("val")
        if not val:
            raise Exception("Invalid val: {}".format(val))

        return val


if __name__ == "__main__":
    TriggerDemo().start()
