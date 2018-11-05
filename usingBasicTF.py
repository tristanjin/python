import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt

num_points = 1000
vectors_set = []

grade = pd.read_csv("./grade.csv")
x_data = grade["high_GPA"]
y_data = grade["univ_GPA"]

x_data = x_data.values
y_data = y_data.values

plt.plot(x_data, y_data, 'ro', label='Original data')
plt.legend()
plt.show()


W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(1000):
    sess.run(train)
    if (step%100)==0:
        print(step, sess.run(W), sess.run(b),sess.run(loss))
        plt.plot(x_data, y_data, 'ro')
        plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
        plt.legend()
        plt.show()
