import queue
import warnings

# from flask import Flask
from wly.work_cpu import CpuThread
from wly.work_gpu import GpuThread
from wly.work_oth import DownThread, PullThread, PushThread

warnings.filterwarnings('ignore')

# app = Flask(__name__)


# http://192.168.1.222:9595/api/gpu/next?modality=CT&st=5

def hello():
    print('Wan Li Yun Provide Services \n Dete Servcie ')


def run():
    hello()
    global que_get, que_pre, que_det, que_ret
    que_get = queue.Queue()
    que_pre = queue.Queue()
    que_det = queue.Queue()
    que_ret = queue.Queue()

    pull_thread = PullThread(que_get, que_pre, que_det, que_ret)
    pull_thread.setDaemon(True)
    pull_thread.start()

    down_thread_nums = 2
    for i in range(down_thread_nums):
        down_thread = DownThread(que_get, que_pre)
        down_thread.setDaemon(True)
        down_thread.start()
    #
    # cpu_thread_nums = 2
    # for i in range(cpu_thread_nums):
    #     cpu_thread = CpuThread(que_pre, que_det, i)
    #     cpu_thread.setDaemon(True)
    #     cpu_thread.start()


    cpu_thread = CpuThread(que_pre, que_det)
    cpu_thread.setDaemon(True)
    cpu_thread.start()


    gpu_thread = GpuThread(que_det, que_ret)
    gpu_thread.setDaemon(True)
    gpu_thread.start()

    pull_thread = PushThread(que_ret)
    pull_thread.setDaemon(True)
    pull_thread.start()

    pull_thread.join()
    # app.run()
