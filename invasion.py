# -*- coding: utf-8 -*-
"""
Created on 13 Feb 2017

@author: islah
"""
import scipy as sp
import numpy as np
import bwfpnm as bpnm
from bwfpnm import routine_pore_percolation_new as bwfr
import time
import os
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
ctrl = bpnm.Base.Controller()
ctrl.loglevel = 40

loc = os.getcwd() + '/'
# loc = '/home/islah/Documents/01_Year-1/10_Paper2s/01Single/data/'

# %% === Load data and define parameters  ===6
net = 'sintered_glass'
# net = 'CB_7mm_1.2um_1_1800.raw'

# Scale can be changed as in Paper of Islahuddin
scale = 1
direction = 'x'

# for different cases see the paper of Islahuddin
case = 'wetting'  # wetting or imbibition
# trapping = air entrapment
trapping = True
# dp True calculates both the wetting and the drying path, dp false calculates only the wetting path
dp = True
drying_case = 'drying'  # desorption or drying

saveparaview = False
Amethod = 'g'

netsplit = net.split('_')
file_data = os.path.join(loc, net, net + '_BC.p')

loc_props = os.getcwd() + '/'
loc_props += 'bwfpnm_parameters.dat'
phase_props = bpnm.Utilities.IO.load_properties(loc_props)

netinput, geoinput, geomacro = bpnm.Utilities.IO.load_data(file_data)
L = geomacro['size']
netinput.update(macro_Lx=L[0], macro_Ly=L[1], macro_Lz=L[2])
geoinput.update(Amethod=Amethod)

# %% === Create all necessary objects ===
pn = bwfr.network(bpnm.Network.Topology, netname='net', **netinput)
geo = bwfr.geometry(pn, bpnm.Geometry.Topology, geoname='geo', **geoinput)
geo.count_shape()

water, vapour, moisture = bwfr.phase(pn, props=phase_props)

phys_water, phys_vapour, phys_moisture = bwfr.physic(
    pn, water, vapour, moisture)

# Trimm all isolated clusters (has no inlet nor outlet pores)
span, dead, isolated = pn.cluster_types()
try:
    trim_pores = np.hstack(isolated)
    # print(len(trim_pores))
    pn.trim(pores=trim_pores)  # geometry, phase, and physics are adjusted too!
except:
    pass

# %% === Run the invasion algorithm with the desired parameters ===
inv_points = None
# inv_points = -sp.logspace(9, 2, 500)

# % Boundary pores
# ----------------------
# inlet: inlet pores for the invading fluid
# outlet: outlet pores fot the defending fluid
# wetting = adsorption, drying = desorption/drying depend on the BC
# drying vs desorption: see the BC positions
#
#               Adsorption      Desorption      Imbibition      Drying
# Invading      vapour          air             liq water       air
# Defending     air             liq/vap water   air             liq/vap air
#
#           Adsorption      Desorption      Imbibition      Drying
# Inlet     all BC pores*       inlet           inlet
# Outlet    all BC pores*   all BC pores*   outlet          inlet*
#
# *) automatic, arg isn't needed
pinlet = pn['pore.inlet']  # inlet for imbibition & drying
outlet_imb = pn['pore.outlet']  # for imbibition
if case == 'imbibition':
    dp = False
#    trapping = True
elif case == 'wetting':
    pinlet = ~pn['pore.internal']
elif drying_case == 'desorption':
    pinlet = ~pn['pore.internal']  # inlet for desorption

atime = time.time()
alg_wp, alg_dp = bwfr.algorithm(pn, water, vapour, cases=[case],
                                trapping=trapping, dp=dp,
                                inlet_imb=pinlet, inlet_dry=pinlet,
                                outlet=outlet_imb, inv_points=inv_points)
time_alg = time.time() - atime
print('Elapsed time: ' + str(time_alg) + ' CPU seconds')

# %% === Save the algorithm object to .pnm file ===
if case == 'wetting':
    if dp:
        case += '_' + drying_case
elif case == 'imbibition':
    if trapping:
        case += '_AE'

loc = os.getcwd() + '/'
name = loc + net + '_' + case

name += '_' + direction

if Amethod == 'vl':
    name += '_VL'

ctrl.save(name)
# ctrl.export(pn, name)

"""
filename_pores = 'percol_pores_results.csv'
filename_throats = 'percol_throats_results.csv'
dict_pores = dict()
dict_throats = dict()

for key in alg_wp.keys():
    if key.split('.')[0] == 'pore':
        dict_pores.update({key:alg_wp[key]})
    else:
        dict_throats.update({key:alg_wp[key]})

bpnm.Utilities.IO.save_data_csv(data=dict_pores, filename=filename_pores)
bpnm.Utilities.IO.save_data_csv(data=dict_throats, filename=filename_throats)
"""
