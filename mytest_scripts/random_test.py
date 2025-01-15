import random

x0 = int((16 - 4) * random.random())
y0 = int((16 - 4) * random.random())

print (x0)
print(y0)


for i in range(10):
    print(i)
    print(random.random()) # 0-1
    indices = random.sample(range(0, 8), k=2) # what is 
    print(indices)
