import time

def counter():
    count = 0
    while count < 3:
        count += 1
        time.sleep(1)
        print(count, end='\r')

counter()

# Doesn't work on IDE...