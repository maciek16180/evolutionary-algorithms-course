import numpy as np
from es import es_plus, es_comma
from problems import griewank, rastrigin, schwefel, ackeley, sphere
from problems import griewank_init, rastrigin_init, schwefel_init, ackeley_init, sphere_init
from time import time

path = 'zad1/'
log_interval = 20

#print 'Griewank_plus\n\n'
#t0 = time()
#sol, log = es_plus(200, griewank, 20000, 10000, .8, 20000, griewank_init, 
#                   logging=True, log_interval=log_interval)
#np.savez(path + 'griewank_plus', sol, log)
#print '\n', time() - t0, '\n\n'

print 'Rastrigin_plus\n\n'
t0 = time()
sol, log = es_plus(20, rastrigin, 200000, 150000, .2, 15000, rastrigin_init,
                   logging=True, log_interval=log_interval, init_std=.1)
np.savez(path + 'rastrigin_plus', sol, log)
print '\n', time() - t0, '\n\n'

print 'Schwefel_plus\n\n'
t0 = time()
sol, log = es_plus(50, schwefel, 100000, 80000, .1, 15000, schwefel_init,
                   logging=True, log_interval=log_interval, init_std=.1)
np.savez(path + 'schwefel_plus', sol, log)
print '\n', time() - t0, '\n\n'

#print 'Ackeley_plus\n\n'
#t0 = time()
#sol, log = es_plus(50, ackeley, 80000, 50000, .2, 20000, ackeley_init,
#                   logging=True, log_interval=log_interval)
#np.savez(path + 'ackeley_plus', sol, log)
#print '\n', time() - t0, '\n\n'

#print 'Sphere_plus\n\n'
#t0 = time()
#sol, log = es_plus(200, sphere, 20000, 10000, .2, 20000, sphere_init,
#                   logging=True, log_interval=log_interval, init_std=.1)
#np.savez(path + 'sphere_plus', sol, log)
#print '\n', time() - t0, '\n\n'




#print 'Griewank_comma\n\n'
#t0 = time()
#sol, log = es_comma(200, griewank, 20000, 10000, 30000, .8, 20000, griewank_init, 
#                    logging=True, log_interval=log_interval)
#np.savez(path + 'griewank_comma', sol, log)
#print '\n', time() - t0, '\n\n'

print 'Rastrigin_comma\n\n'
t0 = time()
sol, log = es_comma(20, rastrigin, 200000, 150000, 300000, .2, 15000, rastrigin_init,
                    logging=True, log_interval=log_interval, init_std=.1)
np.savez(path + 'rastrigin_comma', sol, log)
print '\n', time() - t0, '\n\n'

print 'Schwefel_comma\n\n'
t0 = time()
sol, log = es_comma(50, schwefel, 100000, 80000, 160000, .1, 15000, schwefel_init,
                    logging=True, log_interval=log_interval, init_std=.1)
np.savez(path + 'schwefel_comma', sol, log)
print '\n', time() - t0, '\n\n'

#print 'Ackeley_comma\n\n'
#t0 = time()
#sol, log = es_comma(50, ackeley, 80000, 50000, 150000, .2, 20000, ackeley_init,
#                    logging=True, log_interval=log_interval)
#np.savez(path + 'ackeley_comma', sol, log)
#print '\n', time() - t0, '\n\n'

#print 'Sphere_comma\n\n'
#t0 = time()
#sol, log = es_comma(200, sphere, 20000, 10000, 30000, .2, 20000, sphere_init,
#                    logging=True, log_interval=log_interval, init_std=.1)
#np.savez(path + 'sphere_comma', sol, log)
#print '\n', time() - t0, '\n\n'
