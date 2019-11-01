import os
import time
from send_message import send
from get_webtime import get_webservertime

def duty_check():
    while 1:
        year, month, day, hour, minute, second = get_webservertime('www.baidu.com').split(' ')
        f = open('duty.txt', 'r').read().split('\n')
        [date, name_list] = f[1].split(' ')
        if [month, day] == date.split('-'):
            name_list = eval(name_list)
            for name in name_list:
                send(name)
            f.remove(f[1])
            g = open('temp.txt', 'w')
            for i in f:
                g.write(i)
                g.write('\n')
            g.close()
            os.remove('duty.txt')
            os.rename('temp.txt', 'duty.txt')
            time.sleep(604800)