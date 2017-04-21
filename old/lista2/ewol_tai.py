import numpy as np
import time
from qap_tools import prepare_qap, plot_qap, qap_eval
from sga import sga

def solve_qap(fname, num_iter, N, M, tc=.9, tm=.9, log_interval=1, plot_file=None):
    dist, flow, opt = prepare_qap(fname)

    globals()[fname] = {'dist': dist, 'flow': flow, 'opt': opt, 'num_iter': num_iter}

    t0 = time.time()
    sol, log = sga(eval(fname)['dist'].shape[0], lambda x: -qap_eval(x, eval(fname)['dist'], eval(fname)['flow']), 
                   N, M, tc, tm, num_iter, logging=True, log_interval=log_interval)
    eval(fname)['sol'] = sol
    eval(fname)['log'] = log
    print time.time() - t0
    
    plot_qap(eval(fname)['log'], eval(fname)['opt'], fname=plot_file)


solve_qap('tai50a', 2, 16000, 12000, .3, 1, log_interval=100, plot_file='tai50a.png')
solve_qap('tai60a', 20000, 16000, 12000, .3, 1, log_interval=100, plot_file='tai60a.png')
solve_qap('tai80a', 20000, 16000, 12000, .3, 1, log_interval=100, plot_file='tai80a.png')


