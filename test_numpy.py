import numpy as np

arr1 = np.arange(0, 3)
arr2 = np.asarray((0.0, 1.0, 2.4, 3.6, 10, 4.9))

print(arr1)
print(arr2)
idx = np.where(np.isclose(arr1, arr2, rtol=0.1))
print(idx)
for i in idx:
    print(arr1[i])
    print(arr2[i])
