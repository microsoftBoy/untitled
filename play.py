import os
import time


def play():
    print('开始游戏')
    for i in range(0, 130):
        os.popen('adb -s M960BDPT22CRU shell input tap 488 942')
        time.sleep(0.1)
        print("i = %i" % i)

    time.sleep(3)
    # 再玩一次
    os.popen('adb -s M960BDPT22CRU shell input tap 412 1948')
    print('再玩一次')
    # 玩游戏
    # os.popen('adb -s M960BDPT22CRU shell input tap 471 1879')

    time.sleep(1.5)
    # OK 我知道了
    os.popen('adb -s M960BDPT22CRU shell input tap 739 1695')
    print('OK 我知道了')
    time.sleep(0.5)
    play()


# 开始游戏
os.popen('adb -s M960BDPT22CRU shell input tap 739 1998')
print('开始游戏 ============')
time.sleep(2)
# OK 我知道了
os.popen('adb -s M960BDPT22CRU shell input tap 739 1695')
print('OK 我知道了 ========')
time.sleep(1)
play()
