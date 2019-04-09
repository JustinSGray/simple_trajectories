from sympy import *

x_start0, x_start1, y_start0, y_start1, x_end0, x_end1, y_end0, y_end1, speed0, speed1, t = symbols('x_start0 x_start1 y_start0 y_start1 x_end0 x_end1 y_end0 y_end1 speed0 speed1 t')


m0 = (y_end0 - y_start0) / (x_end0 - x_start0)
m1 = (y_end1 - y_start1) / (x_end1 - x_start1)

b0 = y_start0 - m0 * x_start0
b1 = y_start1 - m1 * x_start1

x0 = x_start0 + cos(atan(m0)) * speed0 * t
x1 = x_start1 + cos(atan(m1)) * speed1 * t

y0 = m0 * x0 + b0
y1 = m1 * x1 + b1


dist_sq = (x0 - x1)**2 + (y0 - y1)**2

dt = diff(dist_sq, t)

print(dist_sq)