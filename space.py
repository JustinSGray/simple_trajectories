import numpy as np
from openmdao.api import Group, Problem, ExplicitComponent

from itertools import combinations

class Space(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('r_space', default=100.0, types=float)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input(name='x', val=np.zeros(num_nodes), units='m')
        self.add_input(name='y', val=np.zeros(num_nodes), units='m')

        self.add_output(name='err_space_dist', val=np.ones(num_nodes))

        ar = np.arange(num_nodes)
        self.declare_partials('err_space_dist', ['x', 'y'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        num_nodes = self.options['num_nodes']
        r_s = self.options['r_space']

        x = inputs['x']
        y = inputs['y']

        nm = np.sqrt(x**2 + y**2)

        outputs['err_space_dist'] = nm - r_s

    def compute_partials(self, inputs, partials):
        num_nodes = self.options['num_nodes']
        r_s = self.options['r_space']

        x = inputs['x']
        y = inputs['y']

        nm = np.sqrt(x**2 + y**2)

        partials['err_space_dist', 'x'] = x/nm
        partials['err_space_dist', 'y'] = y/nm


if __name__ == '__main__':
    from openmdao.api import Problem, Group
    np.random.seed(1)
    num_nodes = 10
    p = Problem()
    p.model = Group()
    p.model.add_subsystem('test', Space(num_nodes=num_nodes),
                                          promotes=['*'])
    p.setup()

    p['x'] = np.random.uniform(0, 1, num_nodes)
    p['y'] = np.random.uniform(0, 1, num_nodes)

    p.run_model()
    print(p['err_space_dist'])

    #quit()

    p.check_partials(compact_print=True)




