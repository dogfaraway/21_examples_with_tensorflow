import tensorflow as tf

a = tf.random_normal((100, 100))
b = tf.random_normal((100, 500))
c = tf.matmul(a, b)  # 矩阵相乘
sess = tf.InteractiveSession()
sess.run(c)