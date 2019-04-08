import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def intersect(line1, line2, r_space):
    si = []
    for line in [line1, line2]:
        xs, ys, xe, ye = line

        m = (ye - ys)/ (xe - xs)
        b = ys - m*xs

        si.append([m, b])

    m0, b0 = si[0]
    m1, b1 = si[1]
    x_i = (b0 - b1) / (m1 - m0)
    y_i = m0*x_i + b0

    d = np.sqrt(x_i**2 + y_i**2)

    if d < r_space:
        return True, (x_i, y_i)
    else:
        return False, (x_i, y_i)

if __name__ == '__main__':
    
    r_space = 1.0
    num_lines = 10

    np.random.seed(10)

    lines = []
    n_nocross = 0
    for i in range(num_lines):

        theta = np.random.uniform(0, 2*np.pi)
        xs, ys = np.cos(theta), np.sin(theta)
        eps = np.random.uniform(-np.pi/4, np.pi/4)
        xe, ye = np.cos(np.pi+theta + eps), np.sin(np.pi+theta + eps)
        lines.append([xs, ys, xe, ye])
        plt.plot([xs, xe], [ys, ye])

    k = 0
    for line1, line2 in combinations(lines, 2):
        crossed, pt = intersect(line1, line2, r_space)
        if crossed:
            plt.plot(*pt, 'ko')
        else:
            n_nocross += 1
        k += 1
    print("%2.2f%% don't cross" % (100* n_nocross / k))

    circle = plt.Circle((0, 0), r_space, fill=False)
    plt.gca().add_artist(circle)


    plt.tight_layout(pad=1)
    plt.axis('equal')
    plt.xlim(-r_space, r_space)
    plt.ylim(-r_space, r_space)


    plt.show()
