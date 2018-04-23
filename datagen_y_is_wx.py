import numpy as np

np.random.seed(2)

#X1 = np.random.randint(low=1, high=100, size=100)
#X2 = np.random.randint(low=1, high=100, size=100)
X1 = np.random.randint(low=1, high=50000, size=10000)
#X2 = np.random.randint(low=1, high=200, size=100)

print("X1.shape is ",X1.shape)

#Y = 3*X1 + 2*X2 + 4
#Y = 3*X1 + 2*X2 + 1

Y = 3*X1 + 2

print("Y.shape is ",Y.shape)

Z = np.column_stack((X1,Y))
#Z = np.column_stack((X1,X2,Y))

print("Z.shape is ",Z.shape)

np.savetxt("foo.csv", Z , fmt='%i', delimiter=',', comments="")
