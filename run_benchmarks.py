import os
from benchmarks import make_benchmark

try:
    os.remove('bench.txt')
except:
    pass
with open("bench.txt", 'a') as f:
    f.write("n_traj, agg, coloring_time, opt_time, compute_total_time, major_iters, n_constr, n_col_fwd, n_col_rev\n")

for n_traj in [5, 10, 20, 40, 80, 100, 200]:

    col_time, opt_time, deriv_time, n_constr, mi, color_counts = make_benchmark(n_traj)

    with open("bench.txt", 'a') as f:
        f.write("%d, %d, %f, %f, %f, %d, %d, %d, %d\n" % (n_traj, 0, col_time[0], opt_time[0], deriv_time[0], mi[0], n_constr[0], color_counts[0][0], color_counts[0][1]))
        f.write("%d, %d, %f, %f, %f, %d, %d, %d, %d\n" % (n_traj, 1, col_time[1], opt_time[1], deriv_time[1], mi[1], n_constr[1], color_counts[1][0], color_counts[1][1]))




