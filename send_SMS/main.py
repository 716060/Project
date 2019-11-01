import time
import threading
from duty_check import duty_check
from get_webtime import get_webservertime

def watch():
    while 1:
        year, month, day, hour, minute, second = get_webservertime('www.baidu.com').split(' ')
        print(year, month, day, hour, minute, second)
        os.system('adb shell input keyevent KEYCODE_BACK')
        time.sleep(600)

def watch_time():
    global task_list, year, month, day, hour, minute, second
    task_list = [0, 0, 0]
    year, month, day, hour, minute, second = get_webservertime('www.baidu.com').split(' ')
    watch_thread = threading.Thread(target=watch)
    task_thread = threading.Thread(target=duty_check)
    watch_thread.start()
    task_thread.start()

watch_time()
