import os
from benchmarks import make_benchmark

try:
    os.remove('bench.txt')
except:
    pass
with open("bench.txt", 'a') as f:
    f.write("agg, sparsity_time, coloring_time, opt_time, deriv_time, jac_shape, jac_fwd_color, jac_rev_color, sep_const_size, sep_fwd_color, sep_rev_color \n")

cases = [5, 10, 20, 40, 80, 100, 200]
cases = [80]

for n_traj in cases:

    sparsity_times, col_times, opt_time, deriv_time, n_constr, jac_shape, jac_fwd_color, jac_rev_color, sep_cont_sizes, color_counts = make_benchmark(n_traj)

    with open("bench.txt", 'a') as f:
        # print(sparsity_times, col_times, opt_time, deriv_time, n_constr, jac_shape, jac_fwd_color, jac_rev_color, sep_cont_sizes, color_counts, file=f)

        for i in range(2): 
            line = f'{i}, {sparsity_times[i]}, {col_times[i]}, {opt_time[i]}, {deriv_time[i]}, {jac_shape[i]}, {jac_fwd_color[i]}, {jac_rev_color[i]}, {sep_cont_sizes[i]}, {color_counts[i][0]}, {color_counts[i][1]} \n'
            f.write(line)
        


