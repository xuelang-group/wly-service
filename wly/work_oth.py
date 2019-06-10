import os
import shutil
import threading
import time
from functools import partial
from multiprocessing import Pool

import requests

import numpy as np
from wly.utils.ret_utils import error_info, success_ret_info


class DownThread(threading.Thread):
    def __init__(self, que_get, que_pre):
        threading.Thread.__init__(self)
        self.que_get = que_get
        self.data_path = './data/'
        self.que_pre = que_pre
        self.pool = Pool(processes=5)

    def run(self):
        while True:
            try:
                if self.que_get.qsize() > 0:
                    # apply_async  us one cpu core
                    s_time = time.time()
                    result_dict = self.que_get.get(timeout=3)
                    series = result_dict['series']
                    windC = [int(ser['windowCenter'].split('\\')[0]) < 0 for ser in series]
                    if sum(windC):
                        ser = series[windC.index(True)]
                        s_path = os.path.join(self.data_path, ser['seriesUid'])
                        if os.path.exists(s_path):
                            shutil.rmtree(s_path)
                            os.mkdir(s_path)
                        else:
                            os.mkdir(s_path)
                        # down_pool = Pool(processes=5)
                        # for i, file in enumerate(ser['files']):
                        #     # img_name = os.path.join(s_path, file['imageUid'])
                        #     # downloading(file['url'], img_name)
                        #     img_name = os.path.join(s_path, file['imageUid'])

                        self.pool.map(partial(downloading, path=s_path), ser['files'])
                        # down_pool.close()
                        # down_pool.join()
                        result_dict['seriesUid'] = ser['seriesUid']
                        result_dict['data_path'] = s_path
                        result_dict['down_time'] = time.time()
                        self.que_pre.put(result_dict)
                        time.sleep(0.2)
                    else:
                        assert False, 101
                else:
                    time.sleep(1)
                # no find lung windows
            except Exception as e:
                print(" download data error {}".format(e))
                error_info(101, result_dict)
                continue


def downloading(file, path):
    save_path = os.path.join(path, file['imageUid'])
    r = requests.get(file['url'], timeout=3)
    assert r.status_code == 200, 111
    with open(save_path + '.dcm', "wb") as code:
        code.write(r.content)



class PullThread(threading.Thread):
    def __init__(self, que_get, que_pre, que_det, que_ret):
        threading.Thread.__init__(self)
        self.que_get = que_get
        self.que_pre = que_pre
        self.que_det = que_det
        self.que_ret = que_ret
        self.pull_data_url = 'http://39.96.243.14:9191/api/gpu/next?modality=CT&st=5'

    def run(self):
        while True:
            que_gsize = self.que_get.qsize()
            que_psize = self.que_pre.qsize()
            que_dsize = self.que_det.qsize()
            que_rsize = self.que_ret.qsize()
            try:
                if que_dsize < 4 and que_gsize < 2 and que_rsize < 2 and que_psize < 3:
                    status, req_result = pull_task_http(self.pull_data_url)
                    assert status, 110
                    if status:
                        if req_result['errCode'] == 0:
                            val = req_result['val']
                            assert not val == {}
                            result_dict = {}
                            result_dict['studyInstanceUid'] = val['studyInstanceUid']
                            result_dict['customStudyInstanceUid'] = val['customStudyInstanceUid']
                            result_dict['series'] = val['series']
                            self.que_get.put(result_dict)
                            time.sleep(0.2)
                else:
                    print('{}- {} - {} - {} '.format(que_gsize, que_psize, que_dsize, que_rsize))
                    time.sleep(1)
            except Exception as e:
                print('PULL ERROR: {}'.format(e))
                # time.sleep(1)
                # continue


def pull_task_http(pull_data_url):
    result = requests.get(pull_data_url, timeout=3)
    if result.status_code == 200:
        result_json = result.json()
        return True, result_json
    else:
        return False, {}


class PushThread(threading.Thread):
    def __init__(self, que_ret):
        threading.Thread.__init__(self)
        self.que_ret = que_ret
        self.push_data_url = 'http://39.96.243.14:9191/api/gpu/submit'

    def run(self):
        while True:
            que_rsize = self.que_ret.qsize()
            if que_rsize > 0:
                try:
                    result_dict = self.que_ret.get()
                    json_info = success_ret_info(result_dict)
                    url = self.push_data_url + '?customStudyUid=' + result_dict['customStudyInstanceUid']
                    result = requests.post(url, json_info, timeout=2)
                    result_json = result.json()
                    print(' {} : {} '.format(result_json, json_info))
                except Exception as e:
                    print(e)
                    error_info(110, result_dict)
            else:
                time.sleep(1)
