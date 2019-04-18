import numpy as np

from openmdao.api import ExplicitComponent, Group, DirectSolver
from constraint_aggregator import ConstraintAggregatorCol
from plane import PlanePath2D
from dymos import ODEOptions
from space import Space
from sum_comp import SumComp
from pairwise import Pairwise



def make_ODE(n_traj, r_space, min_sep, agg):

    class PlaneODE2D(Group):
        ode_options = ODEOptions()

        ode_options.declare_time(units='s')

        # dynamic trajectories
        for i in range(n_traj):
            ode_options.declare_state(name='p%dx' % i, rate_source='p%d.x_dot' % i, 
                                      targets=['p%d.x' % i, 
                                               'pairwise.x%d' % i,
                                               'space%d.x' % i], units='m')
            ode_options.declare_state(name='p%dy' % i, rate_source='p%d.y_dot' % i, 
                                      targets=['p%d.y' % i, 
                                               'pairwise.y%d' % i,
                                               'space%d.y' % i], units='m')
            # ode_options.declare_state(name='p%dmass' % i, rate_source='p%d.mass_dot' % i, 
            #                           targets=['p%d.mass' % i], units='kg')
            # ode_options.declare_state(name='p%dimpulse' % i, rate_source='p%d.impulse_dot' % i,
            #                            targets=['t_imp.a%d' % i])

            ode_options.declare_parameter(name='speed%d' % i, 
                                          targets='p%d.speed' % i, 
                                          dynamic=True)
            ode_options.declare_parameter(name='heading%d' % i, 
                                          targets='p%d.heading' % i, 
                                          dynamic=False)
            ode_options.declare_parameter(name='isp%d' % i, 
                                          targets='p%d.isp' % i, 
                                          dynamic=False)


        def initialize(self):   
            self.options.declare('num_nodes', types=int)
            self.options.declare('r_space', types=float, default=r_space)
            self.options.declare('ignored_pairs', types=list, default=[])

        def setup(self):
            nn = self.options['num_nodes']
            r_space = self.options['r_space']
            pairs = self.options['ignored_pairs']


            sub = self.add_subsystem('sub', Group(), promotes=['*'])

            for i in range(n_traj):
                sub.add_subsystem(name='p%d' % i,
                                   subsys=PlanePath2D(num_nodes=nn))

                sub.add_subsystem(name='space%d' % i,
                                   subsys=Space(num_nodes=nn, 
                                                r_space=r_space))

            sub.add_subsystem(name='pairwise', 
                               subsys=Pairwise(n_traj = n_traj,
                                               ignored_pairs=pairs, 
                                               num_nodes = nn,
                                               min_sep=1.5*min_sep))

            sub.linear_solver = DirectSolver()
            
            if agg:
                n_pairs = n_traj * (n_traj - 1) // 2
                self.add_subsystem(name='agg', 
                                   subsys=ConstraintAggregatorCol(
                                                             c_shape=(n_pairs, nn), 
                                                             rho=50.0,
                                                             nn = nn,
                                                             aggregator='KS'))
                self.connect('pairwise.dist', 'agg.g')

    return PlaneODE2D

