import time

wait_time = 30

start = time.time()
diff = time.time() - start

print('Waiting time begins.....')
while(diff < wait_time):
    diff = time.time() - start

