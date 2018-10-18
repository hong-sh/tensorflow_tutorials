import tensorflow as tf
tf.enable_eager_execution()

tfe = tf.contrib.eager #Shorthand for some symbols

from math import pi
import numpy

def f(x) :
    return tf.square(tf.sin(x))

assert f(pi/2).numpy() == 1.0

# grad_f will return a list of derivatives of f
# with respect to its arguments. Since f() has a single argument, 
# grad_f will return a list with a single element.

grad_f = tfe.gradients_function(f)
assert tf.abs(grad_f(pi/2)[0]).numpy() < 1e-7

def grad(f) :
    return lambda x : tfe.gradients_function(f)(x)[0]

x = tf.lin_space(-2*pi, 2*pi, 100) # 100 points between 2ㅠ and +2ㅠ

import matplotlib.pyplot as plt

plt.plot(x, f(x), label="f")
plt.plot(x, grad(f)(x), label="first derivative")
plt.plot(x, grad(grad(f))(x), label = "second derivative")
plt.plot(x, grad(grad(grad(f)))(x), label = "third derivative")
plt.legend()
plt.show()


def f(x, y) :
    output = 1
    # Must use range(int(y)) instead of range(y) in Python 3 when
    # using TensorFlow 1.10 and earlier. Can use range(y) in 1.11+
    for i in range(int(y)) :
        output = tf.multiply(output, x)
    return output

def g(x, y) :
    # Return the gradient of 'f' with respect to it's first parameter
    return tfe.gradients_function(f)(x, y)[0]

assert f(3.0, 2).numpy() == 9.0 # f(x, 2) is essentially x * x
assert g(3.0, 2).numpy() == 6.0 # And its gradient will be 2 * x
assert f(4.0, 3).numpy() == 64.0 # f(x, 3) is essentially x * x* x
assert g(4.0, 3).numpy() == 48.0 # And its gradient will be 3 * x * x

x = tf.ones((2, 2))

# a single t.gradient() call when the bug is resolved.
with tf.GradientTape(persistent = True) as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# Use the same tape to compute the derivative of z with respect to the
# intermediate value y.
dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0
print(dz_dy.numpy())

# Derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x)
for i in [0,1] :
    for j in [0, 1]:
        assert dz_dx[i][j].numpy() == 8.0
        print(dz_dx.numpy())

