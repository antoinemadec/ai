#!/usr/bin/env python3

import numpy as np
from imread import imread
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

# vector: rank 1, shape (3,)
a = np.array([1, 2, 3])
print(type(a), a.shape, a[0], a[1], a[2])

# matrix: rank 2, shape(2,3)
b = np.array([[1, 2, 3], [4, 5, 6]])
print(type(b), b.shape, b[0, 0], b[0, 1], b[1, 0])

# usefull matrices
print(np.zeros((2, 2)))
print(np.ones((2, 2)))
print(np.full((2, 2), 7))
print(np.eye(2))
print(np.random.random((2, 2)))

# indexing
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
b = a[:2, 1:3]
print(b)
b[0, 0] = 77
print(a[0, 1])
row_r1 = a[1, :]    # rank 1 view of the second row of a
row_r2 = a[1:2, :]  # rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # prints "[[5 6 7 8]] (1, 4)"
# -- integer indexing
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a[[0, 1, 2], [0, 1, 0]])  # prints [1 4 5]
a[np.arange(3), [0, 1, 0]] += 10
print(a)
# -- boolean array indexing
print(a[a > 10])

# math
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
# -- elementwise
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(np.sqrt(x))
# -- mat mult
print(np.dot(x, y))
# -- reduction
print(np.sum(x))
print(np.sum(x, axis=0))
# --transpose
print(x.T)

# broadcasting: when matrix/vector shape don't match
v = np.array([1, 2, 3])
w = np.array([4, 5])
print(np.reshape(v, (3, 1)) * w)
x = np.array([[1, 2, 3], [4, 5, 6]])
print(x+v)
print(x + np.reshape(w, (2, 1)))
print(x*2)

# scipy
# -- matlab: io.loadmat, io.savemat
# -- distance between points
x = np.array([[0, 1], [1, 0], [2, 0]])
# compute the euclidean distance between all rows of x.
# d[i, j] is the euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
d = squareform(pdist(x, 'euclidean'))
print(d)

# matplotlib
x = np .arange(0, 3*np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
# --subplot
img = imread('./cat.jpg')
img_tinted = img * [0.7, 1, 0.7]
# grid is (1,2), first is active
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
# a slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. to work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()
