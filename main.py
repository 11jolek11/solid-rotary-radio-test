from collections import deque
from multiprocessing import Process

from backend import PowerThread
import time
import matplotlib.pyplot as plt
import datetime
from scipy.signal import find_peaks
import numpy as np


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

        for item in history.items():
            try:
                idx = data["x"].index(item[1][0])
                if idx in history.keys():
                    history[item[0]][1].append(data["y"][idx])
            except IndexError:
                continue
        if 99 in history.keys():
            temp = list(history[99][1])[-10::]
            x = np.arange(0, len(temp))
            y = np.array(temp)
            z = np.polyfit(x, y, 1)
            print("{0}x + {1}".format(*z))

        # data["x"]: list of numpy float64
        plt.plot(data["x"], data["y"])
        plt.vlines(peaks_x, ymin=-120, ymax=10, colors="r")
        plt.scatter(peaks_x, peaks_y, c="g")
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%H_%M_%S')
        plt.savefig(f"./img/spec_{formatted_time}__{current_time}.png")
        plt.clf()

    time.sleep(2)
    time.sleep(1)
    print("DIE")
    t.die()
    thread.kill()

