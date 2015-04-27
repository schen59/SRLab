__author__ = 'shaofeng'

import time

def profiler(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print "\nFinish execution in %s seconds" % (end - start)
        return result
    return wrapper