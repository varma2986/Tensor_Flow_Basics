import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)
seed=1
learning_rate = 0.01
num_epochs = 100000
#num_epochs = 1000000

# Normalize all of the features so that they're on the same numeric scale.
# Not doing this can lead to errors in the training process.
def normalize_features(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/ sigma

def shuffle_in_unison(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(len(a))
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

X_Y_train = np.loadtxt('foo.csv', delimiter=',')
#X_train = X_Y_train[:,0]
#Y_train = X_Y_train[:,1]
#X_train = np.loadtxt('foo.csv', delimiter=',', usecols=(0), unpack=True)
#Y_train = np.loadtxt('foo.csv', delimiter=',', usecols=(1), unpack=True)

X_train = np.random.randint(low=1, high=200, size=70)
Y_train = 3*X_train*X_train
#X_train = np.linspace(-3, 3, 100)
#Y_train = np.sin(X_train) + np.random.uniform(-0.5, 0.5, 100)

X_train = np.float32(X_train)
Y_train = np.float32(Y_train)

print("X_train shape is",X_train.shape)
print("Y_train shape is",Y_train.shape)

m = X_train.shape[0]
#n = X_train.shape[1]

#normalize_features(X_train)
#normalize_features(Y_train)
shuffle_in_unison(X_train,Y_train)

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

#W1 = tf.get_variable("W1", shape=[1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
#W2 = tf.get_variable("W2", shape=[1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
#W3 = tf.get_variable("W3", shape=[1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
#W4 = tf.get_variable("W4", shape=[1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
#W5 = tf.get_variable("W5", shape=[1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
#b = tf.get_variable("b", shape=[1], initializer=tf.contrib.layers.xavier_initializer(seed=0))

b = tf.Variable(tf.random_normal([1]), name='bias')
W1 = tf.Variable(tf.random_normal([1]), name='weight1')
W2 = tf.Variable(tf.random_normal([1]), name='weight2')
#W3 = tf.Variable(tf.random_normal([1]), name='weight3')
#W4 = tf.Variable(tf.random_normal([1]), name='weight4')
#W5 = tf.Variable(tf.random_normal([1]), name='weight5')

Y_pred =  tf.multiply(tf.pow(X,2),W2) + tf.multiply(tf.pow(X,1),W1) + b
#Y_pred = tf.multiply(tf.pow(X,5),W5) + tf.multiply(tf.pow(X,4),W4) + tf.multiply(tf.pow(X,3),W3) + tf.multiply(tf.pow(X,2),W2) + tf.multiply(tf.pow(X,1),W1) + b

cost = tf.reduce_mean(tf.square(Y - Y_pred))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-02).minimize(cost) 
#optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost) 

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
costs = []

min_cost = 0
min_cost_epoch = 0
temp_cost = 0
for epoch in range(num_epochs):
    for i in range(m):
        sess.run(optimizer, feed_dict={X: X_train[i], Y: Y_train[i]})        
    temp_cost = sess.run(cost, feed_dict={X:X_train,Y:Y_train})
    costs.append(temp_cost)
    print("Epoch %d Cost:%f"%(epoch,temp_cost))
    if epoch%10==0:
        print("Epoch %d Cost:%f"%(epoch,temp_cost))
    if epoch==1:
        min_cost = temp_cost
        min_cost_epoch = epoch
    if temp_cost<min_cost:
        min_cost = temp_cost
        min_cost_epoch = epoch

plt.plot(costs)
plt.savefig('costs.png')
#print("W1 is %f",sess.run(W1))
#print("W2 is %f",sess.run(W2))
#print("W3 is %f",sess.run(W3))
#print("b1 is ",sess.run(b1))
#print("b2 is ",sess.run(b2))
#print("b3 is ",sess.run(b3))
print("min_cost is ",min_cost)
print("min_cost_epoch is ",min_cost_epoch)
#sess.close()

#print("X is")
#for i in range(m):
#    print(X_train[i])

