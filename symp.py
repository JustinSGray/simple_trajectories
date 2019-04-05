from sympy import *


inputs = 'x y vx vy heading g isp thrust mass CD'

# ----------------
outputs = {}
inputs_unpacked = ','.join(inputs.split())
exec('%s = symbols("%s")' % (inputs_unpacked, inputs))
exec('input_symbs = [%s]' % inputs_unpacked)
# -----------------

outputs['mass_dot'] = -sqrt(thrust**2)/( g * isp)


# ------------------
for oname in outputs:
    print()
    for iname in input_symbs:
        deriv = diff(outputs[oname], iname)
        if deriv != 0:
            st = "\t\tpartials['%s', '%s'] = %s" % (oname, iname, deriv)
            print(st)