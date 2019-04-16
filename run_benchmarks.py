import os
from benchmarks import make_benchmark

try:
    os.remove('bench.txt')
except:
    pass
with open("bench.txt", 'a') as f:
    f.write("n_traj, agg, opt_time, compute_total_time\n")


for n_traj in [5, 10]:

    opt_time, deriv_time = make_benchmark(n_traj)

    with open("bench.txt", 'a') as f:
        f.write("%d, %d, %f, %f\n" % (n_traj, 0, opt_time[0], deriv_time[0]))
        f.write("%d, %d, %f, %f\n" % (n_traj, 1, opt_time[1], deriv_time[1]))




