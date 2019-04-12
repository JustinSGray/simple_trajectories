import numpy as np
from aggregator_funcs import aggf, transform
from openmdao.api import Group, Problem, ExplicitComponent

class ConstraintAggregator(ExplicitComponent):
    """
    Transform and aggregate constraints to a single value.
    """
    def initialize(self):
        self.options.declare('aggregator', types=str)
        self.options.declare('rho', default=50.0, types=float)
        self.options.declare('reversed', default=False, types=bool)
        self.options.declare('c_shape', types=tuple)

    def setup(self):
        agg = self.options['aggregator']
        shape = self.options['c_shape']
        self.reversed = False
        if self.options['reversed']:
            self.reversed = True
        self.aggf = aggf[agg]

        self.add_input(name='g', val=np.ones(shape))
        self.add_output(name='c', val=1.0)

        self.declare_partials('c', 'g')

    def compute(self, inputs, outputs):
        rho = self.options['rho']
        shape = self.options['c_shape']
        g = inputs['g']

        scale = 1.0
        if self.reversed:
            scale = -1.0
        k, dk = self.aggf(scale*g.flatten(), rho)

        outputs['c'] = np.sum(scale*k)
        self.dk = dk.reshape(shape)

    def compute_partials(self, inputs, partials):
        partials['c', 'g'] = self.dk


if __name__ == '__main__':
    from openmdao.api import Problem, Group
    np.random.seed(1)
    shape = (5,5)
    m = 0.0

    p = Problem()
    p.model = Group()
    p.model.add_subsystem('test', ConstraintAggregator(
                                          reversed=True,
                                          rho=2.0,
                                          c_shape=shape,
                                          aggregator='RePU'), promotes=['*'])
    p.setup()
    g = np.random.uniform(-1, 1, shape)
    p['g'] = g

    p.run_model()

    print(p['c'])

    p.check_partials(compact_print=False)