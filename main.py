from collections import deque
from threading import Thread
from multiprocessing import Process

from matplotlib.cbook import index_of
from backend import PowerThread
import time
import matplotlib.pyplot as plt
import datetime
from scipy.signal import find_peaks
import numpy as np
import itertools


# if __name__ == "__main__":
#     t = PowerThread()
#     t.setup(start_freq=433, stop_freq=434, bin_size=10)
#     
#     print("Start new thread")
#     thread = Thread(target=t.run)
#     thread.start()
#     print("Booted")
#
#     trt = None
#
#     while not trt:
#         #time.sleep(2)
#         #print(t.get_history())
#         if len(t.get_history()) >= 1:
#             trt = t.get_history()[0][0]
#         t.select_target(trt)
#
#     time.sleep(2)
#     print(t.get_target_data())
#     time.sleep(1)
#     print("DIE")
#     t.die()
#     thread.join()

if __name__ == "__main__":
    t = PowerThread()
    t.setup(start_freq=433, stop_freq=434, bin_size=10)
    
    print("Start new thread")
    thread = Process(target=t.run)
    thread.start()
    print("Booted")

    trt = None
    active = True

    history = {}

    while active:

        data = t.get_data()
        peaks, _ = find_peaks(data["y"], distance=300, height=-40)

        peaks_x = []
        peaks_y = []

        for index in peaks:
            peaks_x.append(data["x"][index])
            peaks_y.append(data["y"][index])

            if index not in history.keys():
                history[index] = list([None, None])
                history[index][0] = data["x"][index]
                history[index][1] = deque([data["y"][index]], maxlen=4000)
            # if index in history.keys():
            #     history[index][0] = data["x"][index]
            #     history[index][1].append(data["y"][index])
            # else:
            #     history[index][0] = data["x"][index]
            #     history[index][1] = deque([data["y"][index]], maxlen=400)

        for item in history.items():
            try:
                # print(">> ", end=" ")
                # print(item)
                idx = data["x"].index(item[1][0])
                if idx in history.keys():
                    history[item[0]][1].append(data["y"][idx])
            except IndexError:
                continue
        # print(history)
        # if 99 in history.keys():
        #     temp = list(history[99][1])[-10::]
        #     # print(temp)
        #     x = np.arange(0, len(temp))
        #     y = np.array(temp)
        #     z = np.polyfit(x, y, 1)
        #     print("{0}x + {1}".format(*z))

        # data["x"]: list of numpy float64
        plt.plot(data["x"], data["y"])
        plt.vlines(peaks_x, ymin=-120, ymax=10, colors="r")
        plt.scatter(peaks_x, peaks_y, c="g")
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%H_%M_%S')
        plt.savefig(f"./img/spec_{formatted_time}__{current_time}.png")
        plt.clf()

    time.sleep(2)
    # print(t.get_target_data())
    time.sleep(1)
    print("DIE")
    t.die()
    thread.kill()

