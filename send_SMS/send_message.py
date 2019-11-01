import os
import time

def send(member=('张翔辰','18060995316')):
    name, number = member
    message = name + ',今天要记得扫地哦！我是智能短信助手小协。'
    print(message)
    os.system(f'adb shell am start -a android.intent.action.SENDTO -d sms:{number} --es sms_body {message}')
    time.sleep(2)
    os.system('adb shell input keyevent 22')
    time.sleep(2)
    os.system('adb shell input keyevent 66')
    print('Sent successfully!')
    os.system('adb shell input keyevent KEYCODE_BACK')