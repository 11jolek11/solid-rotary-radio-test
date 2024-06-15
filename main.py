from threading import Thread
from multiprocessing import Process
from backend import PowerThread
import time


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

    while active:
        # print(">> ")
        #time.sleep(2)
        #print(t.get_history())
        # print(t.get_data())
        temp = t.get_history()
        if len(temp) >= 1:
            trt = temp[0][0]
            print(">> Target")
            active = False
            thread.kill()
        t.select_target(trt)

    time.sleep(2)
    print(t.get_target_data())
    time.sleep(1)
    print("DIE")
    t.die()
    thread.kill()

