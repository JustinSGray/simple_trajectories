import numpy as np
from numpy import cos, sin, sign
from openmdao.api import ExplicitComponent

g = 9.80665

class PlanePath2D(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('heading', val=0.0, units='rad')
        #self.add_input('speed', val=0.0, units='m/s')
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

        self.add_input(name='mass',
                       val=np.ones(nn),
                       desc='mass',
                       units='kg')

        self.add_input(name='speed',
                       val=np.ones(nn),
                       desc='speed',
                       units='m/s')

        # ------------------------------------------------------

        self.add_output(name='x_dot',
                        val=np.zeros(nn),
                        desc='downrange (longitude) velocity',
                        units='m/s')

        self.add_output(name='y_dot',
                        val=np.zeros(nn),
                        desc='crossrange (latitude) velocity',
                        units='m/s')

        self.add_output(name='mass_dot',
                        val=np.zeros(nn),
                        desc='mass rate',
                        units='kg/s')

        self.add_output(name='impulse_dot',
                        val=np.zeros(nn),
                        desc='impulse_dot')

        ar = np.arange(nn)

        self.declare_partials('x_dot', 'heading')
        self.declare_partials('x_dot', 'speed', rows=ar, cols=ar)

        self.declare_partials('y_dot', 'heading')
        self.declare_partials('y_dot', 'speed', rows=ar, cols=ar)

        self.declare_partials('impulse_dot', 'speed')

    def compute(self, inputs, outputs):
        heading = inputs['heading']
        isp = inputs['isp']
        speed = inputs['speed']
        mass = inputs['mass']

        outputs['x_dot'] = np.cos(heading) * speed
        outputs['y_dot'] = np.sin(heading) * speed

        outputs['impulse_dot'] = speed * np.sign(speed) * mass


    def compute_partials(self, inputs, partials):
        heading = inputs['heading']
        isp = inputs['isp']
        speed = inputs['speed']
        mass = inputs['mass']

        partials['x_dot', 'heading'] = -speed*sin(heading)
        partials['x_dot', 'speed'] = cos(heading)

        partials['y_dot', 'heading'] = speed*cos(heading)
        partials['y_dot', 'speed'] = sin(heading)

        #partials['impulse_dot', 'speed'] = mass*sign(speed)
        #partials['impulse_dot', 'mass'] = speed*sign(speed)



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
    p['speed'] = np.random.uniform(0, 1)
    p['mass'] = np.random.uniform(0, 1, n)

    p.run_model()
    p.check_partials(compact_print=True)

