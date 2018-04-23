import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)
seed=1
#learning_rate = 0.001
num_epochs = 100000
X_Y_train = np.loadtxt('foo.csv', delimiter=',')
X_train = X_Y_train[:,0]
Y_train = X_Y_train[:,1]
#X_train = np.loadtxt('foo.csv', delimiter=',', usecols=(0), unpack=True)
#Y_train = np.loadtxt('foo.csv', delimiter=',', usecols=(1), unpack=True)

#Convert to matrix
X_train = np.asmatrix(X_train)
Y_train = np.asmatrix(Y_train)

X_train = X_train.T
Y_train = Y_train.T

X_train = np.float32(X_train)
Y_train = np.float32(Y_train)

print("X_train shape is",X_train.shape)
#print("X_train is",X_train)

print("Y_train shape is",Y_train.shape)
#print("Y_train is",Y_train)

m = X_train.shape[0]
n = X_train.shape[1]

#Y_train = tf.reshape(Y_train,[1,m])
X=tf.placeholder(tf.float32,shape=(m,n))
Y=tf.placeholder(tf.float32,shape=(m,1))

# Set model weights
W=tf.Variable(tf.random_normal([1,n]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')
#b=tf.Variable(tf.random_normal([1,m]), name='bias')

print("Here1")

pred = tf.matmul(X,W) + b
cost = tf.reduce_mean(tf.square(Y - pred))

print("Here2")
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost)

print("Here3")

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
costs = []

for epoch in range(num_epochs):
    _, temp_cost = sess.run([optimizer, cost], feed_dict={X:X_train,Y:Y_train})
    #_, temp_cost = sess.run([optimizer,cost],feed_dict={X:X_train,Y:Y_train})
    costs.append(temp_cost)
    if epoch%10==0:
        print("Epoch %d Cost:%f"%(epoch,temp_cost))

plt.plot(costs)
plt.savefig('costs.png')
print("W is %f"%sess.run(W))
print("b is ",sess.run(b))

#sess.close()
