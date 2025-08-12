import numpy as np
from time import process_time
"""
NumPy is a library in Python
NumPy can have multiple dimensions
Scalar(0D), Vector(1D), Matrix(2D), Tensor(3D)

NumPy aims to provide an array object that is up to 50x faster than traditional Python lists.

NumPy Arrays vs Python Lists
Fixed Size: Arrays have a fixed size, while lists can dynamically grow.
Homogeneous Data: Arrays require uniform data types; lists can store mixed types.
Performance: Arrays are faster due to their optimized implementation.
Memory Efficiency: Arrays use contiguous memory blocks, unlike lists.

"""
python_list = [i for i in range(10000)]
start_time = process_time()
python_list = [i+5 for i in python_list]
end_time = process_time()
print(end_time - start_time)

np_array = np.array([i for i in range(10000)])
start_time = process_time()
np_array += 5
end_time = process_time()
print(end_time - start_time)

#creating a n dim array
a = np.array([1,2,3,4])
b = np.array([(1,2,3,4),(5,6,7,8)])
print(a.shape)
print(b.shape)
c = np.array([(1,2,3,4),(5,6,7,8)], dtype=float)
print(c)

#initial placeholders in numpy arrays
x = np.zeros((4,5))
print(x)

y = np.ones((3,3))
print(y)

z = np.full((5,4), 5)
print(z)

#create identity matrix
a = np.eye(4)
print(a)

#create numpy array with random values
b = np.random.random((3,4))
print(b)

#random integer values with specific range
c = np.random.randint(0,100,(3,3))
print(c)

#array of evenly spaced values with number of values
d = np.linspace(10,30,6)
print(d)

#array of evenly spaced values with step
e = np.arange(10,30,5)
print(e)

#convert list to numpy array
list = [10,20,30]
f = np.asarray(list)
print(f)

#Analyze numpy array
c = np.random.randint(10,90,(5,5))
print(c)
print(c.shape)
print(c.ndim)

#number of elements in array
print(c.size)

#what type of each element
print(c.dtype)

#mathematical operations
list1 = [1,2,3,4,5]
list2 = [6,7,8,9,10]
print(list1 + list2) #concantenate

a = np.random.randint(0,10,(3,3))
b = np.random.randint(10,20,(3,3))
print(a)
print(b)

print(a+b)
print(a*b)
print(a-b)
print(a/b)

print(np.add(a,b))
print(np.subtract(a,b))

#reshaping
a = np.random.randint(0,10,(2,3))
print(a.shape)
print(a)
b = a.reshape(3,2)
print(b)




