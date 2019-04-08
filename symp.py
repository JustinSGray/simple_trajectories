from sympy import *


inputs = 'x0 x1 y0 y1'

# ----------------
outputs = {}
inputs_unpacked = ','.join(inputs.split())
exec('%s = symbols("%s")' % (inputs_unpacked, inputs))
exec('input_symbs = [%s]' % inputs_unpacked)
# -----------------

outputs['dist'] = sqrt((x0 - x1)**2 + (y0 - y1)**2)


# ------------------
for oname in outputs:
    print()
    for iname in input_symbs:
        deriv = diff(outputs[oname], iname)
        if deriv != 0:
            st = "\t\tpartials['%s', '%s'] = %s" % (oname, iname, deriv)
            print(st)