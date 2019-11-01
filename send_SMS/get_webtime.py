import http.client
from datetime import datetime, timezone, timedelta

def get_webservertime(host):
    try:
        Month_dict = {'Mon': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                      'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        conn = http.client.HTTPConnection(host)
        conn.request("GET", "/")
        r = conn.getresponse()
        ts = r.getheader('date')
        ts = ts[:3] + ts[4:]
        weekday, day, month, year, hh_mm_ss, _ = ts.split(' ')
        hour, minute, second = hh_mm_ss.split(':')
        day, year, hour, minute, second = map(int, [day, year, hour, minute, second])
        month = Month_dict[month]
        utc_time = datetime(year, month, day, hour, minute, second)
        cst_time = utc_time.astimezone(timezone(timedelta(hours=16))).strftime("%Y %m %d %H %M %S")
        return cst_time
    except:
        print('Error: unable to get time')