import numpy as np

x = np.array([0.0, 0.0, 1.0, 0.0, 0.0])

w = np.array([0.5, -0.2, 1.3, 0.1, -0.7])
b = 0.4

print("input  x =", x)
print("weight w =", w)
print("bias   b =", b)
print()

print("term by term:")
total = 0.0
for i in range(len(x)):
    term = w[i] * x[i]
    total += term
    print(f" w[{i}] * x[{i}] = {w[i]:+.1f} * {x[i]:+.1f} = {term:+.2f}")
print(f" bias = {b:+.2f}")
total += b
print(f" total = {total:+.2f}")
print()

print("one line with numpy:")
print(" np.dot(w, x) + b =", np.dot(w, x) + b)
print()
print("One weight per input. One bias per neuron.")
