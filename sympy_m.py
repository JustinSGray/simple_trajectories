from sympy import *

x, y, t, ts, a, m = symbols('x y t ts a m')

dist = (tanh(t - ts) + 1.0) / 2.0

print(diff(dist, t))