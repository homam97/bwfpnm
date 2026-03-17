import numpy as np
import scipy as sp
from ._qupc import qupc_initialize, qupc_reduce, qupc_update
from numba import jit, njit

"""
File name: find_trapped_pores.py
"""

def _find_trapped_pores(inv_seq, indices, indptr, outlets, vol):
    Np = len(inv_seq)
    sorted_seq = np.vstack((inv_seq.astype(np.int_), np.arange(Np, dtype=np.int_))).T
    sorted_seq = sorted_seq[sorted_seq[:, 0] != -1]
    sorted_seq = sorted_seq[np.lexsort((sorted_seq[:,1], -sorted_seq[:,0]))]
    
    cluster = -np.ones(Np, dtype=np.int_)
    trapped_pores = np.zeros(Np, dtype=np.bool_)
    trapped_clusters = np.ones(Np, dtype=np.bool_)
    trapped_clusters[outlets] = False        
    trapped_steps = -np.ones(Np, dtype=int)
    cluster_map = np.arange(Np, dtype=np.int_)
    next_cluster_num = 0 
    vol_counter = np.zeros(Np)
    vol_trapped = 0
    vol_total = np.sum(vol)
    i = -1
    sat = 0
    for step, pore in sorted_seq:
        i += 1
        n = indices[indptr[pore]:indptr[pore+1]]
        
        if np.all(cluster[n] == -1):
            nc = cluster_map[n][inv_seq[n] >= step]
        else:
            nc = cluster_map[cluster[n]][inv_seq[n] >= step]

        nc_uniq = np.unique(nc)
        if nc.size == 0:
            cluster[pore] = next_cluster_num
            if pore in outlets:
                trapped_clusters[next_cluster_num] = False
                vol_counter[pore] = vol[pore]
            else:  # Otherwise note this cluster as being a trapped cluster
                trapped_clusters[next_cluster_num] = True
                trapped_pores[pore] = True
                trapped_steps[pore] = step
                vol_trapped += vol[pore]
            next_cluster_num += 1

        elif nc_uniq.size == 1:
            c = nc_uniq[0]
            cluster[pore] = c

            if pore in outlets:
                trapped_clusters[c] = False
                cluster_map = qupc_reduce(cluster_map)
                hits = np.where(cluster_map == cluster_map[c])[0]
                trapped_clusters[hits] = False
                vol_counter[pore] = vol[pore]
            
            if trapped_clusters[c]:
                trapped_pores[pore] = True
                trapped_steps[pore] = step
                vol_trapped += vol[pore]

            else:
                vol_counter[pore] = vol[pore]

        elif nc_uniq.size > 1:
            cluster[pore] = min(nc_uniq)
            for c in nc:
                qupc_update(cluster_map, c, min(nc_uniq))
            cluster_map = qupc_reduce(cluster_map)
            if np.all(trapped_clusters[nc]):
                trapped_pores[pore] = True
                trapped_steps[pore] = step
                vol_trapped += vol[pore]
            else:  # Otherwise set all neighbor clusters to untrapped!
                trapped_clusters[nc] = False
                vol_counter[pore] = vol[pore]
        
    return trapped_pores, trapped_steps, vol_trapped, sorted_seq[:,1]



