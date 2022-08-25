# Numpy Giriş

import numpy as np

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []

for i in range(len(a)):
    ab.append(a[i] * b[i])

print(ab)

# Numpy Array

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])

a * b

# Create Numpy Array

c = np.array([1, 2, 3, 4, 5])
type(c)

np.zeros(10, dtype=int)

# Choose random 10 numbers between 0 and 10
np.random.randint(0, 10, size=10)

# 10 is mean, std is 4 and shape is (3,4)
np.random.normal(10, 4, (3, 4))

# Ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size=5)

a.ndim
a.shape
a.size
a.dtype

# Reshaping

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)

ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)

# Index selection
a = np.random.randint(10, size=10)
a[0]
a[0:5]

a[0] = 999
a

m = np.random.randint(10, size=(3, 5))

m[0, 0]
m[1, 1]
m[2, 3] = 999
m[2, 3] = 2.9  # numpy keep numbers as one type ex :int or float
m[:, 0]
m[1, :]  # m[row,column]

# Fancy Index

v = np.arange(0, 30, 3)
v[1]
v[4]

catch = [1, 2, 3]
v[catch]

# Conditional transactions

t = np.array([1, 2, 3, 4, 5])

# classical loop
ab = []

for i in t:
    if i < 3:
        print(i)

# With numpy

t < 3

t[t < 3]
t[t > 3]
t[t != 3]
t[t == 3]
t[t >= 3]
t[t <= 3]

# Mathematical Transactions

z = np.array([1, 2, 3, 4, 5])

z / 5
z * 5 / 10
z ** 2
z - 1

np.subtract(z, 1) # to be permanent assign this to a variable p = np.subtract(z,1)
np.add(z, 1)
np.mean(z)
np.sum(z)
np.min(z)
np.max(z)
np.var(z)

# Equation solution with two unknowns

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]])  # coefficients
b = np.array([12, 10])  # results

np.linalg.solve(a, b)
