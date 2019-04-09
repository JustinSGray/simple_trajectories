import numpy as np
from itertools import combinations
from openmdao.api import ExplicitComponent


class Pairwise(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_traj', types=int)
        self.options.declare('num_nodes', types=int)

    def setup(self):
        n_traj = self.options['n_traj']
        nn = self.options['num_nodes']

        self.n_pairs = n_traj * (n_traj - 1) // 2

        self.i_map = {}
        k = 0
        for i, j in combinations(range(n_traj), 2):
            self.i_map[i, j] = k
            k += 1

        self.add_input(name='time', val=np.zeros(nn))

        for i in range(n_traj):
            self.add_input(name='x%d' % i, val=np.zeros(nn), units='m')
            self.add_input(name='y%d' % i, val=np.zeros(nn), units='m')


        self.add_output(name='dist', val=np.zeros((self.n_pairs, nn)))

        ar = np.arange(nn)

        for i in range(n_traj):
            self.declare_partials('dist', 'x%d' % i)
            self.declare_partials('dist', 'y%d' % i)

    def compute(self, inputs, outputs):
        for i, j in self.i_map.keys():
            k = self.i_map[i, j]

            x0, x1 = inputs['x%d' % i], inputs['x%d' % j]
            y0, y1 = inputs['y%d' % i], inputs['y%d' % j]

            dist = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)

            outputs['dist'][k] = dist

    def compute_partials(self, inputs, partials):
        n_traj = self.options['n_traj']
        nn = self.options['num_nodes']
        for i, j in self.i_map.keys():
            k = self.i_map[i, j]

            x0, x1 = inputs['x%d' % i], inputs['x%d' % j]
            y0, y1 = inputs['y%d' % i], inputs['y%d' % j]

            dist = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)

            partials['dist', 'x%d' % i][k] = (x0 - x1)/dist
            partials['dist', 'x%d' % j][k] = -(x0 - x1)/dist

            partials['dist', 'y%d' % i][k] = (y0 - y1)/dist
            partials['dist', 'y%d' % j][k] = -(y0 - y1)/dist


if __name__ == '__main__':
    from openmdao.api import Problem, Group
    nt = 3
    n = 5

    p = Problem()
    p.model = Group()
    p.model.add_subsystem('test', Pairwise(n_traj = nt, 
                                        num_nodes = n), promotes=['*'])
    p.setup()
    np.random.seed(0)
    for i in range(nt):
        p['x%d' % i] = np.random.uniform(0.01, 500, size=n)
        p['y%d' % i] = np.random.uniform(0.01, 500, size=n)

    p.run_model()


    p.check_partials(compact_print=True)