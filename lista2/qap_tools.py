import numpy as np
import os
import matplotlib.pyplot as plt

data_path = 'qap_data/'


def prepare_qap(fname):
    dist, flow = load_qap(fname)
    sol, _ = load_qap_opt(fname)
    opt_sol = qap_eval(sol[np.newaxis], dist, flow)[0] if sol is not None else None

    print 'Optimal solution for {}: {}'.format(fname, opt_sol if opt_sol is not None else 'unknown')
    
    return dist, flow, opt_sol

def plot_qap(log, opt, fname=None):
    plt.plot(xrange(len(log)), log)
    if opt is not None:
        plt.plot(xrange(len(log)), [-opt]*len(log))
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()

def load_qap(fname):
    with open(data_path + fname + '.dat', 'r') as f:
        n = int(f.readline().split()[0])
        dist = []
        flow = []
        
        f.readline()
        for i in xrange(n):
            dist.append(map(int, f.readline().split()))
            
        f.readline()
        for i in xrange(n):
            flow.append(map(int, f.readline().split()))
        
    return np.array(dist).astype(np.float32), np.array(flow).astype(np.float32)

def load_qap_opt(fname):
    fname = data_path + fname + '.sln'
    if not os.path.isfile(fname):
        return None, None
    with open(fname, 'r') as f:
        opt = int(f.readline().split()[1])
        sol = map(int, f.readline().split())
    return np.array(sol) - 1, np.float32(opt)

def qap_eval(P, dist, flow):
    return dist.ravel().dot(np.swapaxes(flow[:, P], 0, 1)[np.indices(P.shape)[0], P].reshape(P.shape[0], -1).T)
