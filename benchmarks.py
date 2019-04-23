import numpy as np

from airspace_phase import make_ODE
from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver, ScipyKrylov
from dymos import Phase, GaussLobatto
from itertools import combinations
from crossing import intersect
import os
import time
import pickle
# from snopt_parse import parse_SNOPT_MI

np.random.seed(3)

def make_benchmark(n_traj):
    opt_time, deriv_time = [], []
    major_it = []
    n_constr = []
    col_times = []
    sparsity_times = []
    color_counts = []
    jac_shapes = []
    jac_fwd_colors = []
    jac_rev_colors = []
    sep_cont_sizes = []

    n_pairs = n_traj * (n_traj - 1) // 2
    r_space = 100.0
    min_sep = 5.0

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
        if not crosses:
            ignored_pairs.append((i, j))
            print("ignoring pair:", (i,j))

    for agg in [False,]:
        print("Running n=", n_traj, "agg:", agg)
        PlaneODE2D = make_ODE(n_traj, r_space, min_sep, agg)

        p = Problem(model=Group())

        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SNOPT'
        p.driver.options['dynamic_total_coloring'] = True
        #p.driver.set_simul_deriv_color('coloring.json')
        p.driver.opt_settings['Start'] = 'Cold'
        p.driver.opt_settings["Major step limit"] = 2.0 #2.0
        p.driver.opt_settings['Major iterations limit'] = 1000000
        p.driver.opt_settings['Minor iterations limit'] = 1000000
        p.driver.opt_settings['Iterations limit'] = 1000000
        p.driver.opt_settings['Major feasibility tolerance'] = 1.5E-4
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-2
        p.driver.opt_settings['iSumm'] = 6

        if agg:
            ignored_pairs = []

        phase = Phase(transcription=GaussLobatto(num_segments=25, order=3),
                      ode_class=PlaneODE2D,
                      ode_init_kwargs={'ignored_pairs' : ignored_pairs})

        p.model.add_subsystem('phase0', phase)
        if not agg:
            p.model.linear_solver = DirectSolver()
        #p.model.linear_solver = ScipyKrylov(atol=1e-4)


        max_time = 1500.0
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
                                     targets=['p%d.speed' % i], upper=40, lower=0.001,
                                     scaler=1.0)

            phase.add_design_parameter('heading%d' % i, opt=False, val=heading,
                                       units='rad')


        phase.add_objective('time', loc='final', scaler=1.0)
        #phase.add_objective('t_imp.', loc='final', scaler=1.0)

        if agg:
            p.model.add_constraint('phase0.rhs_disc.agg.c',
                                      upper=0.0, scaler=1e-3)
        else:
            p.model.add_constraint('phase0.rhs_disc.pairwise.dist',
                                      upper=0.0, scaler=1e-3)
        print("starting setup()")
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

        print("starting run_driver()")
        t = time.time()
        p.run_driver()
        ct = p.driver._total_coloring._coloring_time
        st = p.driver._total_coloring._sparsity_time
        col_times.append(ct)
        sparsity_times.append(st)



        ot = time.time() - t - ct
        opt_time.append(ot)

        nc = p.driver._cons
        sum_nc = sum([nc[k]['size'] for k in nc])

        n_constr.append(sum_nc)

        if agg:
            varname = 'phase0.rhs_disc.agg.c'
        else: 
            varname = 'phase0.rhs_disc.sub.pairwise.dist'

        sep_cont_sizes.append(nc[varname]['size'])
        
        col_fwd, col_rv = p.driver._total_coloring.get_row_var_coloring(varname)
        color_counts.append([col_fwd, col_rv])

        jac_shapes.append(p.driver._total_coloring._shape)
        jac_fwd_colors.append(p.driver._total_coloring.total_solves(do_fwd=True, do_rev=False))
        jac_rev_colors.append(p.driver._total_coloring.total_solves(do_fwd=False, do_rev=True))

        n_computes = 10
        t = time.time()
        for i in range(n_computes):
            p.compute_totals()
        deriv_time.append((time.time() - t)/n_computes)

        # mi = parse_SNOPT_MI()
        mi = -1
        major_it.append(mi)

    return sparsity_times, col_times, opt_time, deriv_time, n_constr, jac_shapes, jac_fwd_colors, jac_rev_colors, sep_cont_sizes, color_counts


if __name__ == '__main__':

    n = 20
    make_benchmark(n)



