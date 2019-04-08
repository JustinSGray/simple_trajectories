import numpy as np

from airspace_phase import PlaneODE2D, n_traj, r_space
from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver
from dymos import Phase, GaussLobatto
from itertools import combinations

import pickle

np.random.seed(2)

p = Problem(model=Group())

p.driver = pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.options['dynamic_simul_derivs'] = True
#p.driver.set_simul_deriv_color('coloring.json')
#p.driver.opt_settings['Start'] = 'Cold'
p.driver.opt_settings["Major step limit"] = 2.0 #2.0
p.driver.opt_settings['Major iterations limit'] = 1000
p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-3
p.driver.opt_settings['Major optimality tolerance'] = 4.0E-1
p.driver.opt_settings['iSumm'] = 6


phase = Phase(transcription=GaussLobatto(num_segments=20, order=3),
              ode_class=PlaneODE2D)

p.model.add_subsystem('phase0', phase)

max_time = 500.0
start_mass = 25.0

phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(10, max_time))

locs = []

for i in range(n_traj):
    theta = np.random.uniform(0, 2*np.pi)
    heading = theta + np.pi*np.random.uniform(0.5, 1.5)
    start_x = r_space*np.cos(theta)
    start_y = r_space*np.sin(theta)

    end_x = r_space*np.cos(heading)
    end_y = r_space*np.sin(heading)

    locs.append([theta, heading, start_x, start_y])

    phase.set_state_options('p%dx' % i,
                            scaler=0.01, defect_scaler=0.1, fix_initial=True, 
                            fix_final=True)
    phase.set_state_options('p%dy' % i,
                            scaler=0.01, defect_scaler=0.1, fix_initial=True, 
                            fix_final=True)
    phase.set_state_options('p%dmass' % i,
                            scaler=0.01, defect_scaler=0.1, fix_initial=True,
                            lower=0.0)
    phase.set_state_options('p%dimpulse' % i,
                            scaler=0.01, defect_scaler=0.1, fix_initial=True)

    phase.add_control('p%dspeed' % i, rate_continuity=False, units='m/s', 
                      opt=True, upper=2, lower=-2, scaler=1.0)

    # phase.add_path_constraint('p%dmass' % i, 
    #                           constraint_name='mass_positive%d' % i, lower=1.0)

    # phase.add_boundary_constraint('space%d.err_space_dist' % i, 
    #                               constraint_name='space%d_err_space_dist' % i, 
    #                               loc='final', equals=0.0)


    phase.add_design_parameter('heading%d' % i, opt=True, val=heading)


#phase.add_objective('time', loc='final')
phase.add_objective('t_imp.sum', loc='final', scaler=0.1)


p.setup()

phase = p.model.phase0


p['phase0.t_initial'] = 0.0
p['phase0.t_duration'] = max_time/2.0

for i in range(n_traj):
    theta, heading, start_x, start_y = locs[i]
    end_x = r_space*np.cos(heading)
    end_y = r_space*np.sin(heading)

    p['phase0.states:p%dx' % i] = phase.interpolate(ys=[start_x, end_x], nodes='state_input')
    p['phase0.states:p%dy' % i] = phase.interpolate(ys=[start_y, end_y], nodes='state_input')

    p['phase0.states:p%dmass' % i] = phase.interpolate(ys=[start_mass, start_mass], nodes='state_input')
    #p['phase0.states:L%d' % i] = phase.interpolate(ys=[0, 0], nodes='state_input')

p.run_driver()


exp_out = phase.simulate()


import matplotlib.pyplot as plt
circle = plt.Circle((0, 0), r_space, fill=False)
plt.gca().add_artist(circle)

t = exp_out.get_val('phase0.timeseries.time')
print("total time:", t[-1])
data = {'t' : t}
for i in range(n_traj):
    x = exp_out.get_val('phase0.timeseries.states:p%dx' % i)
    y = exp_out.get_val('phase0.timeseries.states:p%dy' % i)
    speed = exp_out.get_val('phase0.timeseries.controls:p%dspeed' % i)
    mass = exp_out.get_val('phase0.timeseries.states:p%dmass' % i)
    imp = exp_out.get_val('phase0.timeseries.states:p%dimpulse' % i)
    heading = exp_out.get_val('phase0.timeseries.design_parameters:heading%d' % i)
    #L = exp_out.get_val('phase0.timeseries.states:L%d' % i)
    #print(np.sum(np.sqrt(x**2 + y**2)))
    print()
    data[i] = {'x' : x, 'y' : y, 'speed' : speed, 'mass' : mass, 
               'heading' : heading, 'imp' : imp}
    plt.plot(x, y, 'gray')
    plt.scatter(x, y, cmap='Greens', c=t)

# with open('sim_implicit.pkl', 'wb') as f:
#     pickle.dump(data, f)
plt.tight_layout(pad=1)
plt.axis('equal')
plt.xlim(-r_space, r_space)
plt.ylim(-r_space, r_space)


plt.figure()
plt.suptitle('speed')
plt.subplot(311)
for i in range(n_traj):
    plt.plot(t, data[i]['speed'])

plt.subplot(312)
plt.suptitle('mass')
for i in range(n_traj):
    plt.plot(t, data[i]['mass'])

plt.subplot(313)
plt.suptitle('impulse')
for i in range(n_traj):
    plt.plot(t, data[i]['imp'])


plt.show()






