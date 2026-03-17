# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:17:28 2017

@author: islah
"""
import scipy as sp
import numpy as np
import bwfpnm as bpnm
from bwfpnm import routine_pore_percolation_new as bwfr
import operator as op
import time
from bwfpnm.Algorithms import Permeability
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
ctrl = bpnm.Base.Controller()
ctrl.loglevel = 40

# %% ==== Determine the parameters ====
# net = 'sintered_glass'
net = 'CB_7mm_1.2um_1_1800.raw'
scale = 1  # 1 or 1e-3
case = 'imbibition'  # wetting or imbibition
dp = False
drying_case = 'desorption'  # drying or desorption
direction = 'x'
# setting can be 1,2 or 3 for respectively condensation only,condensation & surface ad/flow or condensation & surface ad/flow & corner ad/flow
setting = 3
loc = os.getcwd() + '/'  # '/home/islah/Documents/01_Year-1/10_Papers/01Single/data/'
Kang = False

trapping = True
knudsen = True
diffusion = True
moist_volume = False
dPc = 1

if setting == 1:  # condensation only
    surface_ad = False
    corner_ad = False
    surf_flow = False
    corner_flow = False
elif setting == 2:  # condensation & surface ad/flow
    surface_ad = True
    corner_ad = False
    surf_flow = True
    corner_flow = True
elif setting == 3:  # condensation & surface ad/flow & corner ad/flow
    surface_ad = True
    corner_ad = True
    surf_flow = True
    corner_flow = True

single_flow = False

savetopo = 1
num_seq = 40
refine = True
dsat_threshold = 0.1
# %% === Load the invasion algorithm object ===
netsplit = net.split('_')
name = net
filename = loc
cases = [case]
if case in ['wetting', 'drying_wetting']:
    trapping = False
    if dp:
        #        filename += name + '_wetting_' + drying_case
        cases = [case, 'drying']
    filename += name + '_wetting_' + drying_case
elif case == 'imbibition':
    filename += name + '_imbibition'
    filename += '_AE'

if scale != 1:
    filename += '_' + str(scale)

filename += '_' + direction
if Kang:
    filename += '_Kang'

print('Load {}'.format(filename))
ctrl.load(filename)

if case in ['wetting', 'drying_wetting']:
    keys = ['net', 'geo', 'water', 'vapour', 'moisture',
            'physics_water', 'physics_vapour', 'physics_moisture',
            'percolation', 'percolation_dp']
    (pn, geo, water, vapour, moisture, phys_water, phys_vapour, phys_moisture,
     alg_wp, alg_dp) = op.itemgetter(*keys)(ctrl)
elif case == 'imbibition':
    keys = ['net', 'geo', 'water', 'vapour', 'moisture',
            'physics_water', 'physics_vapour', 'physics_moisture',
            'percolation']
    (pn, geo, water, vapour, moisture, phys_water, phys_vapour, phys_moisture,
     alg_wp) = op.itemgetter(*keys)(ctrl)
    alg_dp = None

# ctrl.export(network=pn, filename=filename)
pn.cluster_types()
# %% =========== Run the permeability algorithm ============

# case = 'wetting'    # wetting, imbibition, drying_wetting, drying_imbibition
parallel = False  # True -> use PETSc solver
prec = 'gamg'
amg = None  # 'rs'         # [None, 'rs', 'sa', 'ra', 'adaptive']
accel = None  # 'gmres'  # None, 'gmres', 'minimal_residual', 'bicgstab'
umfpack = True
econd = False

save_matrix = False
name_mat = 'netB'
modify_mat = True
row_scaling = False

debug_cluster = False  # False
keep_log = False
perm_error = False  # True:use permeability relative error for stopping criteria
perm_tol = 1e-2
strength = ('symmetric', {'theta': 0.03})
CF = 'RS'
agg = 'standard'
sa_smooth = 'energy'
cycle = 'F'
iter_smooth = 1
if accel is None:
    tol = 1e-10
else:
    tol = 1e-10  # 1e-6
max_level, max_coarse = 30, 300  # 30, 100
maxiter = 1000  # 500
save_grids = False

lr = np.arange(-10, -1)
r = np.lib.scimath.power(10, lr)
pc = -2 * 0.072 / r
pc = sp.int64(pc)
pc_grids = -np.logspace(np.lib.scimath.log10(-pc.min()), np.lib.scimath.log10(-pc.max()), num_seq)
pc_grids = sp.int64(pc_grids)
pcs = pc_grids

# # create moisture content (w) array
alg_wp, alg_dp, w_sat, porosity = bwfr.moist_content(geo, alg_wp,
                                                     water['prop.density'][0],
                                                     alg_dp=alg_dp,
                                                     cases=[case.split('_')[-1]],
                                                     trapping=trapping)

# %% Permeability
atime = time.time()
perm = Permeability(pn, alg_wp, phys_vapour, phys_moisture,
                    alg_dp=alg_dp)
perm.permeability_curve(cases=cases, w_sat=w_sat,
                        trapping=trapping, num_seq=num_seq,
                        knudsen=knudsen, plot=False, printstatus=False,
                        surface_ad=surface_ad, moist_volume=moist_volume,
                        dPc=dPc,
                        diffusion=diffusion, refine=refine, keep_log=keep_log,
                        amg=amg, tol=tol, maxiter=maxiter,
                        max_level=max_level, max_coarse=max_coarse,
                        cycle=cycle, iterations=iter_smooth,
                        strength=strength,
                        CF=CF, agg=agg, smooth=sa_smooth,
                        save_matrix=save_matrix, pcs=pcs, name_mat=name_mat,
                        modify=modify_mat, row_scaling=row_scaling,
                        pc_grids=pcs, debug=debug_cluster,
                        accel=accel, perm_error=perm_error, perm_tol=perm_tol,
                        parallel=parallel, ksp_type='gmres', pc_type=prec,
                        save_grids=save_grids, dsat_threshold=dsat_threshold,
                        single_flow=single_flow, corner_ad=corner_ad,
                        surf_flow=surf_flow, corner_flow=corner_flow, par_pc=0,
                        umfpack=umfpack, econd=econd, save_pcrate=True)
# print("perm: ", perm.permeability_curve['saturation'])


btime = time.time() - atime
print('Elapsed time: ' + str(btime) + ' CPU seconds')
# %% === Save the results into .csv file(s) ======

data = perm.create_data()


if savetopo:
    for case in data.keys():
        filename = loc + net + '_' + case
        if trapping:
            filename += '_AE'
        if scale != 1:
            filename += '_' + str(scale)
            filename += '_sa_kn'
        if refine:
            filename += '_refine'
        if dp:
            filename += '_' + drying_case
        if corner_flow:
            filename += '_SurfFlow'

        filename += '_' + direction
        for temp in data[case].keys():
            if len(data[case][temp]) == 0:
                data[case][temp] = np.zeros_like(data[case]['pc'])
        bpnm.Utilities.IO.save_data_csv(data=data[case], filename=filename)


# %%  === Extrast: PLOT w(pc) and k(pc) in the same figure ===
def _plot_2y(lpc, w, k):
    fig, ax1 = plt.subplots()

    ax1.plot(lpc, w, 'b.')
    ax1.set_xlabel('Capillary pressure [log(-Pa)]')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Moisture content [kg/m3]', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(lpc, k, 'r.')
    ax2.set_ylabel('Permeability', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.show()


# plt.figure()
for case in data.keys():
    x = data[case]['lpc']
    w = data[case]['moisture content']
    km = np.lib.scimath.log10(data[case]['k_moisture'])
    _plot_2y(x, w, km)


# %% Plot hysteresis w(pc), k(pc), and k(w)
if dp:
    lpc1 = data['wetting']['lpc']
    w1 = data['wetting']['moisture content']
    k1 = np.lib.scimath.log10(data['wetting']['k_moisture'])

    lpc2 = data[case]['lpc']
    w2 = data[case]['moisture content']
    k2 = np.lib.scimath.log10(data[case]['k_moisture2'])

    plt.figure('Hysteresis: w(pc)')
    plt.plot(lpc1, w1, '-', lpc2, w2, '--')

    plt.figure('Hysteresis: k(pc)')
    plt.plot(lpc1, k1, '-', lpc2, k2, '--')

    plt.figure('Hysteresis: k(w)')
    plt.plot(w1, k1, '-', w2, k2, '--')

