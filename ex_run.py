import numpy as np

from airspace_phase import PlaneODE2D, n_traj, r_space, min_sep, agg
from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver
from dymos import Phase, GaussLobatto
from itertools import combinations
from crossing import intersect

import pickle

np.random.seed(3)
# seed = 3, nn = 10, segments=25

#19, 15, 28
p = Problem(model=Group())

coloring = True

p.driver = pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.options['dynamic_simul_derivs'] = True
#p.driver.set_simul_deriv_color('coloring.json')
#p.driver.opt_settings['Start'] = 'Cold'
p.driver.opt_settings["Major step limit"] = 2.0 #2.0
p.driver.opt_settings['Major iterations limit'] = 1000000
p.driver.opt_settings['Minor iterations limit'] = 1000000
p.driver.opt_settings['Iterations limit'] = 1000000
p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-8
p.driver.opt_settings['Major optimality tolerance'] = 4.0E-3
p.driver.opt_settings['iSumm'] = 6

locs = []
thetas = np.linspace(0, 2*np.pi, n_traj + 1)
for i in range(n_traj):
    theta = thetas[i]
    heading = theta + np.random.uniform(0.3*np.pi, 1.7*np.pi)
    start_x = r_space*np.cos(theta)
    start_y = r_space*np.sin(theta)

    end_x = r_space*np.cos(heading)
    end_y = r_space*np.sin(heading)

    heading = np.arctan2(end_y - start_y, end_x - start_x)

    locs.append([theta, heading, start_x, start_y, end_x, end_y])

ignored_pairs = []
for i, j in combinations(range(n_traj), 2):
    theta, heading, start_x, start_y, end_x, end_y = locs[i]
    line1 = [start_x, start_y, end_x, end_y]

    theta, heading, start_x, start_y, end_x, end_y = locs[j]
    line2 = [start_x, start_y, end_x, end_y]

    crosses, pt = intersect(line1, line2, r_space)
    if not crosses and coloring:
        ignored_pairs.append((i, j))
        print("ignoring pair:", (i,j))

if not coloring:
    ignored_pairs = []

phase = Phase(transcription=GaussLobatto(num_segments=25, order=3),
              ode_class=PlaneODE2D, 
              ode_init_kwargs={'ignored_pairs' : ignored_pairs})

p.model.add_subsystem('phase0', phase)

max_time = 500.0
start_mass = 25.0

phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(10, max_time))


for i in range(n_traj):

    theta, heading, start_x, start_y, end_x, end_y = locs[i]

    phase.set_state_options('p%dx' % i,
                            scaler=0.1, defect_scaler=0.01, fix_initial=True, 
                            fix_final=False, units='m')
    phase.set_state_options('p%dy' % i,
                            scaler=0.1, defect_scaler=0.01, fix_initial=True, 
                            fix_final=False, units='m')

    phase.add_boundary_constraint('space%d.err_space_dist' % i, 
                                  constraint_name='space%d_err_space_dist' % i, 
                                  loc='final', lower=0.0)


    # phase.add_control('speed%d' % i, rate_continuity=False, units='m/s', 
    #                   opt=True, upper=20, lower=0.0, scaler=1.0)

    phase.add_polynomial_control('speed%d' % i, order=2, units='m/s', opt=True,
                             targets=['p%d.speed' % i], upper=20, lower=0.0, 
                             scaler=1.0)

    phase.add_design_parameter('heading%d' % i, opt=False, val=heading, 
                               units='rad')


phase.add_objective('time', loc='final', scaler=1.0)
#phase.add_objective('t_imp.', loc='final', scaler=1.0)

if agg:
    p.model.add_constraint('phase0.rhs_disc.agg.c', 
                              upper=0.0, scaler=1.0)
else:
    p.model.add_constraint('phase0.rhs_disc.pairwise.dist', 
                              upper=0.0, scaler=1e-3)
p.setup()


phase = p.model.phase0


p['phase0.t_initial'] = 0.0
p['phase0.t_duration'] = max_time/2.0

for i in range(n_traj):
    theta, heading, start_x, start_y, end_x, end_y = locs[i]

    p['phase0.states:p%dx' % i] = phase.interpolate(ys=[start_x, end_x], 
                                                    nodes='state_input')
    p['phase0.states:p%dy' % i] = phase.interpolate(ys=[start_y, end_y], 
                                                    nodes='state_input')

    #p['phase0.states:p%dmass' % i] = phase.interpolate(ys=[start_mass, start_mass], nodes='state_input')
    #p['phase0.states:L%d' % i] = phase.interpolate(ys=[0, 0], nodes='state_input')
import time
t = time.time()
p.run_driver()

exp_out = phase.simulate()
print("optimization time:", time.time() - t)

import matplotlib.pyplot as plt
circle = plt.Circle((0, 0), r_space, fill=False)
plt.gca().add_artist(circle)

t = exp_out.get_val('phase0.timeseries.time')
print("total time:", t[-1])
data = {'t' : t}
for i in range(n_traj):
    x = exp_out.get_val('phase0.timeseries.states:p%dx' % i)
    y = exp_out.get_val('phase0.timeseries.states:p%dy' % i)
    #speed = exp_out.get_val('phase0.timeseries.controls:speed%d' % i)
    #mass = exp_out.get_val('phase0.timeseries.states:p%dmass' % i)
    #imp = exp_out.get_val('phase0.timeseries.states:p%dimpulse' % i)
    heading = exp_out.get_val('phase0.timeseries.design_parameters:heading%d' % i)
    #L = exp_out.get_val('phase0.timeseries.states:L%d' % i)
    #print(np.sum(np.sqrt(x**2 + y**2)))

    xi, yi = p['phase0.states:p%dx' % i], p['phase0.states:p%dy' % i]


    for k in data.keys():
        if k == 't':
            continue
        x2, y2 = data[k]['x'], data[k]['y']
        dist = np.sqrt((x - x2)**2 + (y - y2)**2)
        if dist.min() < min_sep:
            print("!! dist exp check", i, k, dist.min())

        xi2, yi2 =  p['phase0.states:p%dx' % k], p['phase0.states:p%dy' % k]
        dist = np.sqrt((xi - xi2)**2 + (yi - yi2)**2)
        if dist.min() < min_sep:
            print("!! dist impl. check", i, k, dist.min())

    data[i] = {'x' : x, 'y' : y, 
               'heading' : heading, 'loc' : locs[i]}
    theta, heading_, start_x, start_y, end_x, end_y = locs[i]
    idx = np.where(x**2 + y**2 <= r_space**2)

    plt.plot(start_x, start_y, 'ro')
    plt.plot(end_x, end_y, 'bo')

    plt.plot(x[idx], y[idx], 'gray')
    plt.scatter(x[idx], y[idx], cmap='Greens', c=t[idx])

with open('sim_implicit.pkl', 'wb') as f:
    pickle.dump(data, f)
plt.tight_layout(pad=1)
plt.axis('equal')
plt.xlim(-r_space, r_space)
plt.ylim(-r_space, r_space)

n_pairs = n_traj * (n_traj - 1) // 2
print("ignored %d pairs of trajectories out of %d pairs" % (len(ignored_pairs), n_pairs))

# plt.figure()
# plt.suptitle('speed')
# plt.subplot(311)
# for i in range(n_traj):
#     plt.plot(t, data[i]['speed'])

# plt.subplot(312)
# plt.suptitle('mass')
# for i in range(n_traj):
#     plt.plot(t, data[i]['mass'])

# plt.subplot(313)
# plt.suptitle('impulse')
# for i in range(n_traj):
#     plt.plot(t, data[i]['imp'])


plt.show()






