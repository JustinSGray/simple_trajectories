import numpy as np
from numpy import cos, sin
from openmdao.api import ExplicitComponent

g = 9.80665

class PlanePath2D(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('heading', val=0.0)
        self.add_input('isp', val=10.0)

        # ---------------------------------------------------

        self.add_input(name='x',
                       val=np.ones(nn),
                       desc='aircraft position x',
                       units='m')

        self.add_input(name='y',
                       val=np.ones(nn),
                       desc='aircraft position y',
                       units='m')

        self.add_input(name='vx',
                       val=np.ones(nn),
                       desc='aircraft velocity magnitude x',
                       units='m/s')

        self.add_input(name='vy',
                       val=np.ones(nn),
                       desc='aircraft velocity magnitude y',
                       units='m/s')

        self.add_input(name='thrust',
                       val=np.zeros(nn),
                       desc='Thrust',
                       units='N')

        self.add_input(name='mass',
                       val=np.ones(nn),
                       desc='mass',
                       units='kg')

        # ------------------------------------------------------

        self.add_output(name='x_dot',
                        val=np.zeros(nn),
                        desc='downrange (longitude) velocity',
                        units='m/s')

        self.add_output(name='y_dot',
                        val=np.zeros(nn),
                        desc='crossrange (latitude) velocity',
                        units='m/s')

        self.add_output(name='vx_dot',
                        val=np.zeros(nn),
                        desc='accl',
                        units='m/s**2')

        self.add_output(name='vy_dot',
                        val=np.zeros(nn),
                        desc='accl',
                        units='m/s**2')

        self.add_output(name='mass_dot',
                        val=np.zeros(nn),
                        desc='mass rate',
                        units='kg/s')

        self.add_output(name='impulse_dot',
                        val=np.zeros(nn),
                        desc='impulse_dot')

        ar = np.arange(nn)

        self.declare_partials('x_dot', 'vx', rows=ar, cols=ar)
        self.declare_partials('y_dot', 'vy', rows=ar, cols=ar)

        self.declare_partials('mass_dot', 'thrust', rows=ar, cols=ar)

        self.declare_partials('vx_dot', ['thrust', 'mass'], rows=ar, cols=ar)
        self.declare_partials('vx_dot', 'heading')
        self.declare_partials('vy_dot', ['thrust', 'mass'], rows=ar, cols=ar)
        self.declare_partials('vy_dot', 'heading')

        self.declare_partials('impulse_dot', 'thrust', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        heading = inputs['heading']
        isp = inputs['isp']
        vx = inputs['vx']
        vy = inputs['vy']
        thrust = inputs['thrust']
        mass = inputs['mass']

        outputs['x_dot'] = vx
        outputs['y_dot'] = vy

        outputs['mass_dot'] = -np.abs(thrust)/( g * isp)

        mass = np.clip(mass, 1e-8, None)

        outputs['vx_dot'] = (thrust) * cos(heading) / mass
        outputs['vy_dot'] = (thrust) * sin(heading) / mass

        outputs['impulse_dot'] = thrust * np.sign(thrust) * mass


    def compute_partials(self, inputs, partials):
        heading = inputs['heading']
        isp = inputs['isp']
        vx = inputs['vx']
        vy = inputs['vy']
        thrust = inputs['thrust']
        mass = inputs['mass']
        mass = np.clip(mass, 1e-8, None)

        partials['x_dot', 'vx'] = 1.0

        partials['y_dot', 'vy'] = 1.0

        partials['impulse_dot', 'thrust'] = 1.0

        partials['mass_dot', 'thrust'] = -1/(g*isp) * np.sign(thrust)

        partials['vx_dot', 'heading'] = -thrust*sin(heading)/mass
        partials['vx_dot', 'thrust'] = cos(heading)/mass
        partials['vx_dot', 'mass'] = -thrust*cos(heading)/mass**2

        partials['vy_dot', 'heading'] = thrust*cos(heading)/mass
        partials['vy_dot', 'thrust'] = sin(heading)/mass
        partials['vy_dot', 'mass'] = -thrust*sin(heading)/mass**2



if __name__ == '__main__':
    from openmdao.api import Problem, Group

    n = 4

    p = Problem()
    p.model = Group()
    p.model.add_subsystem('test', PlanePath2D(num_nodes = n), promotes=['*'])
    p.setup()

    p['heading'] = 0.5
    p['x'] = np.random.uniform(0, 100, n)
    p['y'] = np.random.uniform(0, 200, n)
    p['vx'] = np.random.uniform(0, 1, n)
    p['vy'] = np.random.uniform(0, 1, n)
    p['thrust'] = np.random.uniform(0, 1, n)
    p['mass'] = np.random.uniform(0, 1, n)

    p.run_model()
    p.check_partials(compact_print=True)

