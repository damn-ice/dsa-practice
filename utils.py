import time
from functools import wraps

def function_time_calculator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        print(f'Time: {time.time() - start}s')
    return wrapper 