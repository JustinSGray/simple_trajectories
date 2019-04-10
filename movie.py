import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import combinations
import os, shutil
import pickle
from airspace_phase import PlaneODE2D, n_traj, r_space

scene = 'implicit'

try:
    shutil.rmtree('frames_%s' % scene)
except FileNotFoundError:
    pass
os.makedirs('frames_%s' % scene)


with open('sim_%s.pkl' % scene, 'rb') as f:
    data = pickle.load(f)


n=len(data['t'])
for t in range(len(data['t']))[::-1]:
    fig = plt.figure()
    ax = plt.gca()
    for i in range(n_traj):
        if data[i]['x'][t]**2 + data[i]['y'][t]**2 > r_space**2:
            continue
        circle = plt.Circle((0, 0), r_space, fill=False)
        plt.gca().add_artist(circle)
        circle = plt.Circle((data[i]['x'][t], data[i]['y'][t]), 10/2, fill=False)
        ax.add_artist(circle)

        plt.title("t = %f" % data['t'][t])
        c =  data['t'][0:t+1]
        s = np.linspace(0.1,1.0, len(c))
        plt.scatter(data[i]['x'][t], data[i]['y'][t], marker='^', cmap='Greens')

        plt.plot(data[i]['x'][:t], data[i]['y'][:t], 'k', linewidth=0.1)

        theta, heading, start_x, start_y, end_x, end_y = data[i]['loc']
        #plt.plot([start_x], [start_y], 's', markersize=6)
        #plt.xlabel('x')
        #plt.ylabel('y')
    #plt.tight_layout(pad=1)
    #plt.axis('equal')
    plt.xlim(-r_space,r_space)
    plt.ylim(-r_space,r_space)
    fig.savefig('frames_%s/%03d.png' % (scene, t), dpi=fig.dpi)

cmd = "ffmpeg -y -r 25 -i frames_%s/%%03d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p out_%s.mp4; open out_%s.mp4" % (scene, scene, scene)
os.system(cmd)

#plt.show()
