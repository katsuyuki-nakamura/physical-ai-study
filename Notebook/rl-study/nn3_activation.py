import numpy as np

rng = np.random.default_rng(0)

x = rng.normal(size=5)
W1 = rng.normal(0.0, 0.5, size=(64, 5))
W2 = rng.normal(0.0, 0.5, size=(2, 64))

two_layers = W2 @ (W1 @ x)
one_layer = (W2 @ W1) @ x

print("--- without activation ---")
print("  two layers    :", np.round(two_layers, 4))
print("  folded to one :", np.round(one_layer, 4))
print("  equal         :", np.allclose(two_layers, one_layer))
print("  -> any number of layers collapses into one. Depth buys nothing.")

with_tanh = W2 @ np.tanh(W1 @ x)

print("--- with tanh ---")
print("  two layers   :", np.round(with_tanh, 4))
print("  same as one  :", np.allclose(with_tanh, one_layer))
print("  -> no. Depth starts to mean something only once a nonlinearity sits in between.")
print()

print("tanh squeezes any value into -1..1:")
for v in (-3.0, -1.0, 0.0, 1.0, 3.0):
    print(f"  tanh({v:+.1f}) = {np.tanh(v):+.4f}")
