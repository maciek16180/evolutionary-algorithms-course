import numpy as np
from es import es, plot_sol
from time import time
from problems import griewank, rastrigin, schwefel, ackeley, michalewicz

path = 'zad1/'

max_it = 30000
dim = 200
pop_size = 10000
num_parents = 5000
k = .8
log_interval = 20

print 'Griewank\n\n'
t0 = time()
sol, log = es(dim, griewank, pop_size, num_parents, k, max_it, logging=True, log_interval=log_interval)
np.savez(path + 'griewank', sol, log)
print '\n', time() - t0, '\n\n'

print 'Rastrigin\n\n'
t0 = time()
sol, log = es(dim, rastrigin, pop_size, num_parents, k, max_it, logging=True, log_interval=log_interval)
np.savez(path + 'rastrigin', sol, log)
print '\n', time() - t0, '\n\n'

print 'Schwefel\n\n'
t0 = time()
sol, log = es(dim, schwefel, pop_size, num_parents, k, max_it, logging=True, log_interval=log_interval)
np.savez(path + 'schwefel', sol, log)
print '\n', time() - t0, '\n\n'

print 'Ackeley\n\n'
t0 = time()
sol, log = es(dim, ackeley, pop_size, num_parents, k, max_it, logging=True, log_interval=log_interval)
np.savez(path + 'ackeley', sol, log)
print '\n', time() - t0, '\n\n'