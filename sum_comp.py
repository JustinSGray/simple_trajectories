
import numpy as np
from openmdao.api import ExplicitComponent

class SumComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_arrays', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        na = self.options['num_arrays']
        ar = np.arange(nn)

        self.add_output(name='sum', val=np.zeros(nn))

        for i in range(na):
            self.add_input(name='a%d' % i,
                           val=np.zeros(nn))

            self.declare_partials('sum', "a%d" % i, rows=ar, cols=ar)

        self.dx = np.ones(nn)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        na = self.options['num_arrays']

        out = np.zeros(nn)
        for i in range(na):
            out += inputs['a%d' % i]

        outputs['sum'] = out

    def compute_partials(self, inputs, partials):
        na = self.options['num_arrays']
        for i in range(na):
            partials['sum', 'a%d' % i] = self.dx


if __name__ == '__main__':
    from openmdao.api import Problem, Group

    nn = 20
    na = 5

    p = Problem()
    p.model = Group()
    p.model.add_subsystem('test', SumComp(num_nodes = nn, num_arrays=na), promotes=['*'])
    p.setup()

    for i in range(na):
        p['a%d' % i] = np.random.uniform(-100, 100, nn)

    p.run_model()
    p.check_partials(compact_print=True)