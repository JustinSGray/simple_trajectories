import numpy as np

import matplotlib.pylab as plt

from mpltools import annotation

# colored data 
n_vehicles = np.array([5,10,20,40,80])
agg=np.array([0,0,0,0,0])
sparsity_time=np.array([0.8871049881,14.02198386,19.56522799,69.407305,453.9343059])
coloring_time=np.array([0.05682706833,0.5786988735,1.913504124,24.94848895,278.910187])
opt_time=np.array([3.043201923,36.26545024,87.00453877,173.2298923,1896.050537])
deriv_time=np.array([0.03581960201,0.4468089819,0.2842278957,0.7366415977,3.003323483])
jac_rows=np.array([756,2761,10521,41041,162081])
jac_cols=np.array([267,532,1062,2122,4242])
jac_fwd_color=np.array([7,9,15,21,29])
jac_rev_color=np.array([1,1,1,1,1])
sep_const_size=np.array([500,2250,9500,39000,158000])
sep_fwd_color=np.array([6,8,14,20,28])
sep_rev_color=np.array([0,0,0,0,0])


jac_colored = jac_rev_color + jac_fwd_color


log_n_vehicles = np.log10(n_vehicles)
log_jac_cols = np.log10(jac_cols)
log_jac_colored = np.log10(jac_colored)

print('None - Aggregated:')
print('non-colored')
fit = np.polyfit(log_n_vehicles, log_jac_cols, 1)
print(fit)

print('colored')
fit = np.polyfit(log_n_vehicles, log_jac_colored, 1)
print(fit)


fig, ax = plt.subplots()
ax.loglog(n_vehicles, jac_cols, lw=5, color='C0')
ax.loglog(n_vehicles, jac_cols, marker='o', markersize=9, color='w', lw=0)
ax.loglog(n_vehicles, jac_cols, marker='o', markersize=5, color='C0', lw=0)

ax.loglog(n_vehicles, jac_colored, lw=5, color='C1')
ax.loglog(n_vehicles, jac_colored, marker='o', markersize=9, color='w', lw=0)
ax.loglog(n_vehicles, jac_colored, marker='o', markersize=5, color='C1', lw=0)


# aggregated data

n_vehicles=np.array([5,10,20,40,80])
agg=np.array([1,1,1,1,1])
sparsity_time=np.array([9.319242001,36.37544084,56.04355478,447.6675589,2136.103796])
coloring_time=np.array([0.1577773094,0.3356459141,0.4127249718,1.113538027,4.171155214])
opt_time=np.array([29.42631555,38.62718821,58.78727698,460.8148639,2304.248504])
deriv_time=np.array([0.08443198204,0.1452581167,0.2115468025,0.4721776009,1.411782479])
jac_rows=np.array([306,561,1071,2091,4131])
jac_cols=np.array([267,532,1062,2122,4242])
jac_fwd_color=np.array([4,4,4,4,4])
jac_rev_color=np.array([4,4,4,4,4])
sep_const_size=np.array([50,50,50,50,50])
sep_fwd_color=np.array([0,0,0,0,0])
sep_rev_color=np.array([3,3,3,3,3])

jac_colored = jac_rev_color + jac_fwd_color


log_n_vehicles = np.log10(n_vehicles)
log_jac_cols = np.log10(jac_cols)
log_jac_colored = np.log10(jac_colored)


print('Aggregated:')
print('non-colored')
fit = np.polyfit(log_n_vehicles, log_jac_cols, 1)
print(fit)

print('colored')
fit = np.polyfit(log_n_vehicles, log_jac_colored, 1)
print(fit)

# fig, ax = plt.subplots()
# ax.loglog(n_vehicles, jac_cols)
ax.loglog(n_vehicles, jac_colored, lw=5, color='C2')
ax.loglog(n_vehicles, jac_colored, marker='o', markersize=9, color='w', lw=0)
ax.loglog(n_vehicles, jac_colored, marker='o', markersize=5, color='C2', lw=0)

annotation.slope_marker((12, 548), 1.0, ax=ax)
annotation.slope_marker((40, 18), 0.5, ax=ax)

ax.text(10,1000,"Non-colored", color='C0')
ax.text(20,24,"Colored", color='C1')
ax.text(25,10,"Aggregated", color='C2')

ax.set_xlabel('Number of Trajectories')
ax.set_ylabel('Number of\nLinear Solves', rotation='horizontal', ha='right', multialignment='center')
ax.set_xticks((5,10,20,40,80))
ax.set_xticklabels((5,10,20,40,80))
fig.savefig('color_scaling.pdf', bbox_inches='tight')

