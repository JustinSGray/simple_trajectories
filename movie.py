import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import combinations
import os, shutil
import pickle

scene = 'implicit'

try:
    shutil.rmtree('frames_%s' % scene)
except FileNotFoundError:
    pass
os.makedirs('frames_%s' % scene)


with open('sim_%s.pkl' % scene, 'rb') as f:
    data = pickle.load(f)

n_traj, r_space, min_sep = data['n_traj'], data['r_space'], data['min_sep']


n=len(data['t'])
for t in range(len(data['t']))[::-1]:
    fig = plt.figure()
    ax = plt.gca()
    for i in range(n_traj):
        x, y = data[i]['x'][t], data[i]['y'][t]
        theta, heading, start_x, start_y, end_x, end_y = data[i]['loc']
        if x**2 + y**2 > r_space**2:
            x, y = end_x, end_y
        circle = plt.Circle((0, 0), r_space, fill=False)
        plt.gca().add_artist(circle)
        circle = plt.Circle((x, y), min_sep/2.3, fill=False)
        ax.add_artist(circle)

        plt.title("t = %f" % data['t'][t])
        c =  data['t'][0:t+1]
        s = np.linspace(0.1,1.0, len(c))
        plt.scatter(x, y, marker='^', cmap='Greens')


        xx = np.linspace(start_x, x, 10)
        yy = np.linspace(start_y, y, 10)
        plt.plot(xx, yy, 'k', linewidth=0.1)

        #plt.plot(data[i]['x'][:t], data[i]['y'][:t], 'k', linewidth=0.1)


        #plt.plot([start_x], [start_y], 's', markersize=6)
        #plt.xlabel('x')
        #plt.ylabel('y')
    plt.tight_layout(pad=1)
    plt.axis('equal')
    plt.xlim(-r_space,r_space)
    plt.ylim(-r_space,r_space)
    fig.savefig('frames_%s/%03d.png' % (scene, t), dpi=fig.dpi)

cmd = "ffmpeg -y -r 50 -i frames_%s/%%03d.png -c:v libx264 -vf fps=50 -pix_fmt yuv420p out_%s.mp4; open out_%s.mp4" % (scene, scene, scene)
os.system(cmd)

#plt.show()
