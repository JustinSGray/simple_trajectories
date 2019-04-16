import os
from benchmarks import make_benchmark

try:
    os.remove('bench.txt')
except:
    pass
with open("bench.txt", 'a') as f:
    f.write("n_traj, agg, coloring_time, opt_time, compute_total_time, major_iters, opt_exit_code\n")

for n_traj in [5, 10, 20, 40, 80, 100, 120, 160, 200, 250, 300, 400, 500, 800, 1000]:

    col_time, opt_time, deriv_time, success, mi = make_benchmark(n_traj)

    with open("bench.txt", 'a') as f:
        f.write("%*d, %d, %*f, %*f, %*f, %*d, %d\n" % (4,n_traj, 0, 15,col_time[0], 15,opt_time[0], 15,deriv_time[0], 5, mi[0], 1*success[0]))
        f.write("%*d, %d, %*f, %*f, %*f, %*d, %d\n" % (4,n_traj, 1, 15,col_time[1], 15,opt_time[1], 15,deriv_time[1], 5, mi[1], 1*success[1]))




