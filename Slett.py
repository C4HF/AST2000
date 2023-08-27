import numpy as np
a = [1,2,3,5,10] 
b = [2,5,6,78,8,9,10]

for i in range(len(a)):
    if a[i] in b:
        b.remove(a[i])
print(b)