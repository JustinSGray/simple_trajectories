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
        self.options.declare('nn', types=int)

    def setup(self):
        agg = self.options['aggregator']
        shape = self.options['c_shape']

        self.nn = self.options['nn']


        self.reversed = False
        if self.options['reversed']:
            self.reversed = True
        self.aggf = aggf[agg]

        self.add_input(name='g', val=np.zeros(shape))
        self.add_output(name='c', val=np.zeros(self.nn))

        self.declare_partials('c', 'g')

    def compute(self, inputs, outputs):
        rho = self.options['rho']
        shape = self.options['c_shape']
        g = inputs['g']

        scale = 1.0
        if self.reversed:
            scale = -1.0

        partials = np.zeros((self.nn, shape[0], self.nn))
        for i in range(self.nn):
            k, dk = self.aggf(scale*g[:,i], rho)
            outputs['c'][i] = np.sum(scale*k)
            partials[i, :, i] = dk

        self.dk = partials.reshape((self.nn, self.nn*shape[0]))

    def compute_partials(self, inputs, partials):
        partials['c', 'g'] = self.dk

class ConstraintAggregatorCol(ExplicitComponent):
    """
    Transform and aggregate constraints to a single value.
    """
    def initialize(self):
        self.options.declare('aggregator', types=str)
        self.options.declare('rho', default=50.0, types=float)
        self.options.declare('reversed', default=False, types=bool)
        self.options.declare('c_shape', types=tuple)
        self.options.declare('nn', types=int)

    def setup(self):
        agg = self.options['aggregator']
        shape = self.options['c_shape']

        self.nn = self.options['nn']


        self.reversed = False
        if self.options['reversed']:
            self.reversed = True
        self.aggf = aggf[agg]

        self.add_input(name='g', val=np.zeros(shape))
        self.add_output(name='c', val=np.zeros(self.nn))

        partials = np.zeros((self.nn, shape[0], self.nn))
        for i in range(self.nn):
            partials[i, :, i] = np.ones(shape[0])

        partials = partials.reshape((self.nn, self.nn*shape[0]))
        self.rows, self.cols = np.where(partials > 0)
        self.declare_partials('c', 'g', rows=self.rows, cols=self.cols)

    def compute(self, inputs, outputs):
        rho = self.options['rho']
        shape = self.options['c_shape']
        g = inputs['g']

        scale = 1.0
        if self.reversed:
            scale = -1.0

        partials = np.zeros((self.nn, shape[0], self.nn))
        for i in range(self.nn):
            k, dk = self.aggf(scale*g[:,i], rho)
            outputs['c'][i] = np.sum(scale*k)
            partials[i, :, i] = dk

        self.dk = partials.reshape((self.nn, self.nn*shape[0]))[self.rows, self.cols]

    def compute_partials(self, inputs, partials):
        partials['c', 'g'] = self.dk

if __name__ == '__main__':
    from openmdao.api import Problem, Group
    np.random.seed(1)
    shape = (5, 10)
    m = 0.0

    p = Problem()
    p.model = Group()
    p.model.add_subsystem('test', ConstraintAggregator(
                                          reversed=True,
                                          rho=2.0,
                                          nn=10,
                                          c_shape=shape,
                                          aggregator='RePU'), promotes=['*'])
    p.setup()
    g = np.random.uniform(-1, 1, shape)
    p['g'] = g

    p.run_model()

    print(p['c'])

    p.check_partials(compact_print=True)