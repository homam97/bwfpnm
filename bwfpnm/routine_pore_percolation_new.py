# -*- coding: utf-8 -*-
"""
Created on Fri May 20 23:56:49 2016

This is a module which consists of routine functions for moisture storage and transfer estimations for topological network model

--> use throat.porelengths in conductivity calculation

Customised for Percolation class

modified routine_pore_percolation:
- algorithm: case -> cases (list of cases)
- permeability: use bwfpnm.Algorithm.GenericMultiscaleLinearTransport (using amg)

"""
import bwfpnm as bpnm
import scipy as sp
import numpy as np
from numpy.linalg import cond

ctrl = bpnm.Base.Controller()
ctrl.loglevel = 40

#%% Network
def network(NetClass=None, netname='net', dat_file=None, trimming=False,
            **kwargs):

    if dat_file is not None:
        netinput, geoinput, geomacro =\
            bpnm.Utilities.IO.load_data(dat_file)

    #% Create network object
    if NetClass is None:
        NetClass = bpnm.Network.Topology

    try:
        pn = NetClass(name=netname, **netinput)
    except:
        pn = NetClass(name=netname, **kwargs)

    #% Trim 'isolated' cluster
    if trimming:
        health = pn.check_network_health()
        trim_pores = health['trim_pores']
        # modify geoinput based on trimpore
        pn.trim_geom_data(geoinput, trim_pores)
        pn.trim(pores=trim_pores)

    try:
        return (pn, geoinput, geomacro)
    except:
        return pn


#%% Geometry
def geometry(NetInstance, GeoClass=None, geoname='geo',
             pores='all', throats='all', geoinput=None, **kwargs):

    ps = NetInstance.pores(pores)
    ts = NetInstance.throats(throats)

    if GeoClass is None:
        try:
            GeoClass = bpnm.Geometry.Topology
        except:
            GeoClass = bpnm.Geometry.Test_PoreThroat

    try:
        geo = GeoClass(network=NetInstance, pores=ps, throats=ts,
                       name=geoname, **geoinput)
    except:
        geo = GeoClass(network=NetInstance, pores=ps, throats=ts,
                       name=geoname, **kwargs)

    return geo


#%% Create phases: water + vapour = moisture
def phase(NetInstance, WaterClass=None, VapourClass=None,
          watername='water', vapourname='vapour', props=None,
          **kwargs):

    if WaterClass is None:
        WaterClass = bpnm.Phases.Water
    if VapourClass is None:
        VapourClass = bpnm.Phases.Vapour

    water = WaterClass(name=watername, network=NetInstance, props=props['water'],
                       **kwargs)
    vapour = VapourClass(name=vapourname, network=NetInstance, props=props['vapour'],
                         **kwargs)
    vapour['prop.water_density'] = water['prop.density']

    moisture = bpnm.Phases.GenericPhase(name='moisture', network=NetInstance,
                                        props=props['moisture'], **kwargs)
#    moisture['prop.temperature'] = temperature
    return (water, vapour, moisture)


#%% Create physics
def physic(NetInstance, WaterInstance, VapourInstance, MoistInstance,
           pores='all', throats='all',
           PhysWaterClass=None, PhysVapourClass=None, PhysMoistClass=None,
           physwatername='physics_water', physvapourname='physics_vapour',
           physmoistname='physics_moisture', param='shapefactor', **kwargs):

    if PhysWaterClass is None:
        PhysWaterClass = bpnm.Physics.Standard_Topology_pore
    if PhysVapourClass is None:
        PhysVapourClass = bpnm.Physics.GenericPhysics
    if PhysMoistClass is None:
        PhysMoistClass = bpnm.Physics.GenericPhysics

    ps = NetInstance.pores(pores)
    ts = NetInstance.throats(throats)
    # The physics for water is calculated beforehand since it is independent of
    # the ambient condition (rh or Pc)
    phys_water = PhysWaterClass(network=NetInstance, phase=WaterInstance,
                                pores=ps, throats=ts, param=param,
                                dynamic_data=True, name=physwatername)
    # The physics of vapour (and hence moisture) is calculated on the fly
    # since it is dependent to the ambient conditin (rh or Pc)
    phys_vapour = PhysVapourClass(network=NetInstance, phase=VapourInstance,
                                  pores=ps, throats=ts, dynamic_data=True,
                                  name=physvapourname)
    phys_moisture = PhysMoistClass(network=NetInstance, phase=MoistInstance,
                                   pores=ps, throats=ts, dynamic_data=True,
                                   name=physmoistname)

    return (phys_water, phys_vapour, phys_moisture)


#%% Create algorithm: Wetting & Drying Percolation -> only use element's Pc_wp
def algorithm(NetIns, WaterIns, VapourIns, Class=None, cases=['wetting'],
              npts=None, dp=False, trapping=False,
              inlet_imb=None, inlet_dry=None, outlet=None,
              name='percolation', **kwargs):

    if Class is None:
        Class = bpnm.Algorithms.Percolation
    alg_wp = Class(network=NetIns, phase_wet=WaterIns, phase_dry=VapourIns,
                   name=name, **kwargs)

    #%% Boundary pores
    # ----------------------
    # inlet: inlet pores for the invading fluid
    # outlet: outlet pores fot the defending fluid
    # wetting = adsorption, drying = desorption/drying depend on the BC
    #
    #               Adsorption      Desorption      Imbibition      Drying
    # Invading      vapour          air             liq water       air
    # Defending     air             liq/vap water   air             liq/vap air
    #
    #           Adsorption      Desorption      Imbibition      Drying
    # Inlet     all BC pores*   all BC pores    inlet           inlet
    # Outlet    all BC pores*   all BC pores*   outlet          inlet*
    #
    # *) automatic, arg isn't needed
    bc_pores = ~NetIns['pore.internal']
    if inlet_imb is None:
        inlet_imb = NetIns['pore.inlet']    # bool
    if outlet is None:
        outlet = NetIns['pore.outlet']
    if inlet_dry is None:
        # for both desorption & drying: inlet = outlet
        # desorption: bc_pores, drying: inlet
        inlet_dry = bc_pores                # desorption
        inlet_dry = NetIns['pore.inlet']    # drying

    for case in cases:
        if case=='wetting':
            # no need to supply the inlet and outlet pores

            alg_wp.run_wetting(**kwargs)
            if trapping:
                outlet = ~NetIns['pore.internal']
                # alg_wp.evaluate_trapping_wetting(p_outlets=outlet, mode='clone')
                alg_wp.evaluate_trapping_wetting_fast(p_outlets=outlet, mode='clone')

                alg_wp.copy_results(prop_names=[case+'_trapped_pc'])
        elif case=='imbibition':
            alg_wp.set_inlets_imbibition(pores=inlet_imb)
            alg_wp.run_imbibition(**kwargs)
            if trapping:
                # alg_wp.evaluate_trapping_imbibition(p_outlets=outlet, mode='clone') 
                alg_wp.evaluate_trapping_imbibition_fast(p_outlets=outlet, mode='clone') 
                
               
    # copy result to wet phase
    alg_wp.copy_results(prop_names=[case+'_inv_seq', case+'_inv_pc',
                                    case+'_inv_sat'])
    if trapping:
        alg_wp.copy_results(prop_names=[case+'_inv_seq_trapping',
                                        case+'_inv_pc_trapping',
                                        case+'_inv_sat_trapping',
                                        # case+'_trapped_seq',
                                        case+'_trapped_pc'])
    # if desorption/drying from wetting/imbibition
    if dp:
        alg_dp = Class(network=NetIns, phase_wet=WaterIns, phase_dry=VapourIns,
                       name='percolation_dp', **kwargs)

        for case in cases:
            if trapping:
                pinv = alg_wp['pore.'+case+'_inv_pc_trapping']
                tinv = alg_wp['throat.'+case+'_inv_pc_trapping']
            else:
                pinv = alg_wp['pore.'+case+'_inv_pc']
                tinv = alg_wp['throat.'+case+'_inv_pc']

#            if alg_wp:
#                if case == 'wetting':
#                    inv_points = alg_wp._wetting_pc
#                elif case == 'imbibition':
#                    inv_points = alg_wp._imbibition_pc

            alg_dp.setup_drying(p_inv_pc=pinv, t_inv_pc=tinv, case=case)
            alg_dp.run_drying(inv_site=inlet_dry, case=case, **kwargs)
            # copy result to wet phase
            case = 'drying_' + case
            alg_dp.copy_results(prop_names=[case+'_inv_seq', case+'_inv_pc',
                                            case+'_inv_sat'])
    else:
        alg_dp = None

    return (alg_wp, alg_dp)


#%% create moisture content (w) array
def moist_content(geo, alg_wp, water_density, cases=['wetting'],
                  trapping=False, alg_dp=None,
                  v_mat=None, porosity=None, **kwargs):

    v_pore = sp.sum(geo['pore.volume']) + sp.sum(geo['throat.volume'])

    if v_mat is None:
        try:
            v_mat = geo._net._macro_Lx * geo._net._macro_Ly * geo._net._macro_Lz
        except:
            v_mat = geo._net._Lx * geo._net._Ly * geo._net._Lz

    porosity = v_pore/v_mat

    try:
        w_sat = porosity*water_density
    except:
        print('error: either volume of bulk material of porosity is required!')
        return


    for case in cases:
        if case.split('_')[0] == 'drying':
            continue
#        case = case.split('_')[-1]
        alg_wp['pore.'+case+'_inv_w'] = alg_wp['pore.'+case+'_inv_sat']*w_sat
        alg_wp['throat.'+case+'_inv_w'] = alg_wp['throat.'+case+'_inv_sat']*w_sat
        if trapping:
            prop = case+'_inv_w_trapping'
            prop_sat = case+'_inv_sat_trapping'
            alg_wp['pore.'+prop] = alg_wp['pore.'+prop_sat]*w_sat
            alg_wp['throat.'+prop] = alg_wp['throat.'+prop_sat]*w_sat

            alg_wp.copy_results(prop_names=[case+'_inv_w_trapping'])
        alg_wp.copy_results(prop_names=[case+'_inv_w'])

    if alg_dp is not None:
        for case in cases:
            case = 'drying_' + case
            prop = case+'_inv_w'
            alg_dp['pore.'+prop] = alg_dp['pore.'+case+'_inv_sat']*w_sat
            alg_dp['throat.'+prop] = alg_dp['throat.'+case+'_inv_sat']*w_sat

            alg_dp.copy_results(prop_names=[case+'_inv_w'])

    return (alg_wp, alg_dp, w_sat, porosity)


#%% Calculate permeability for each Pc value

def permeability(pn, alg_wp, alg_dp, water, vapour, moisture,
                 phys_vapour, phys_moisture, w_sat, case=['wetting'],
                 trapping=False,
                 num_seq=10, knudsen=True, plot=True, printstatus=False,
                 surface_ad=False, moist_volume=False, dPc =1, diffusion=True,
                 **kwargs):
    r'''
    This function has been moved to Algorithm.Permeability.permeability_curve()
    '''

    pm = bpnm.Physics.models
    pab = bpnm.Algorithms

    sat_wp, sat_dp = [], []
    sat_wp_surf, sat_dp_surf = [], []
    sat_wp_moist, sat_dp_moist = [], []
#    max_norm_res = 0
    eff_perm_moisture_wp = {'0': [], '1': [], '2': []}
    eff_perm_moisture_dp = {'0': [], '1': [], '2': []}

    p_volumes = pn['pore.volume']
    t_volumes = pn['throat.volume']
    volume_total = sum(p_volumes) + sum(t_volumes)

    lr = sp.arange(-10, -1)
    r = sp.power(10, lr)
    pc = -2*water['pore.surface_tension'][0]/r

    Pc_min, Pc_max = sp.amin(pc), sp.amax(pc)
    Pc_wp = -sp.logspace(sp.log10(-Pc_min), sp.log10(-Pc_max), num_seq)
#    Pc_wp = sp.around(Pc_wp, 3)
    Pc_dp = Pc_wp[::-1]

    case = case[0]
    occupancy = 'occupancy_'+case
    if trapping:
        occupy = occupancy + '_trapping'
    else:
        occupy = occupancy

    for Pc_step_wp, Pc_step_dp in list(zip(Pc_wp, Pc_wp)):
        alg_wp.return_results(Pc=Pc_step_wp, cases=[case],
                              occupancies=[occupancy], trapping=trapping)
        alg_dp.return_results(Pc=Pc_step_dp, cases=['drying'],
                              occupancies=['occupancy_drying'],
                              trapping=False)
        if (Pc_step_dp+120094) <50000:
            pass

        p_occ_wp = water['pore.'+occupy]
        t_occ_wp = water['throat.'+occupy]
        p_occ_dp = water['pore.occupancy_drying']
        t_occ_dp = water['throat.occupancy_drying']

        volume_p_wp = sum(p_occ_wp*p_volumes)
        volume_t_wp = sum(t_occ_wp*t_volumes)
        volume_p_dp = sum(p_occ_dp*p_volumes)
        volume_t_dp = sum(t_occ_dp*t_volumes)

        saturation_wp = (volume_p_wp + volume_t_wp)/(volume_total)
        saturation_dp = (volume_p_dp + volume_t_dp)/(volume_total)
        sat_wp.append(saturation_wp)
        sat_dp.append(saturation_dp)
        if printstatus:
            print('WP_saturation: %.3f' % saturation_wp,
                  '\t Pc: %.3f' %Pc_step_wp)
            print('DP_saturation: %.3f' % saturation_dp,
                  '\t Pc: %.3f' %Pc_step_dp)
            print('WP_volume: ', sum(p_occ_wp),
                  '\t throat: ', sum(t_occ_wp))
            print('DP_volume: ', sum(p_occ_dp),
                  '\t throat: ', sum(t_occ_dp))
            print('WP_water occupancy: ', p_occ_wp,
                  '\t throat: ', t_occ_wp)
            print('DP_water occupancy: ', p_occ_dp,
                  '\t throat: ', t_occ_dp, '\n')

        # Surface adsorption: p/t.surface_thickness & surface_volume
        if surface_ad:
            phys_vapour.models.add(propname='pore.surface_thickness_wp',
                                   model=pm.surface_adsorption.pstat_thickness,
                                   pc=Pc_step_wp,
                                   pore_occupancy='pore.'+occupy)
            phys_vapour.models.add(propname='throat.surface_thickness_wp',
                                   model=pm.surface_adsorption.tstat_thickness,
                                   pc=Pc_step_wp,
                                   throat_occupancy='throat.'+occupy)
            phys_vapour.models.add(propname='pore.surface_thickness_drying',
                                   model=pm.surface_adsorption.pstat_thickness,
                                   pc=Pc_step_dp,
                                   pore_occupancy='pore.occupancy_drying')
            phys_vapour.models.add(propname='throat.surface_thickness_drying',
                                   model=pm.surface_adsorption.tstat_thickness,
                                   pc=Pc_step_dp,
                                   throat_occupancy='throat.occupancy_drying')

            ## Inner circular tube
            phys_vapour.models.add(propname='pore.surface_volume_wp',
                                   model=pm.surface_adsorption.pvolume,
                                   pore_occupancy='pore.'+occupy,
                                   film_thickness='pore.surface_thickness_wp')
            phys_vapour.models.add(propname='throat.surface_volume_wp',
                                   model=pm.surface_adsorption.tvolume,
                                   throat_occupancy='throat.'+occupy,
                                   film_thickness='throat.surface_thickness_wp')
            phys_vapour.models.add(propname='pore.surface_volume_drying',
                                   model=pm.surface_adsorption.pvolume,
                                   pore_occupancy='pore.occupancy_drying',
                                   film_thickness='pore.surface_thickness_drying')
            phys_vapour.models.add(propname='throat.surface_volume_drying',
                                   model=pm.surface_adsorption.tvolume,
                                   throat_occupancy='throat.occupancy_drying',
                                   film_thickness='throat.surface_thickness_drying')


            volume_p_wp = sum(phys_vapour['pore.surface_volume_wp'])
            volume_t_wp = sum(phys_vapour['throat.surface_volume_wp'])
            volume_p_dp = sum(phys_vapour['pore.surface_volume_drying'])
            volume_t_dp = sum(phys_vapour['throat.surface_volume_drying'])

            sat_surf_wp = (volume_p_wp + volume_t_wp)/(volume_total)
            sat_surf_dp = (volume_p_dp + volume_t_dp)/(volume_total)
            sat_wp_surf.append(sat_surf_wp)
            sat_dp_surf.append(sat_surf_dp)

            sat_wp[-1] += sat_surf_wp   # update total saturations
            sat_dp[-1] += sat_surf_dp

            print('pthickness wp: ',
                  phys_vapour['pore.surface_thickness_wp'][0])
            print('pradius wp: ',
                  pn['pore.diameter'][0]/2)
            print('pvol surf wp: ',
                  phys_vapour['pore.surface_volume_wp'][0])
            print('pvol wp: ',
                  p_volumes[0])
            print('psat wp: ',
                  phys_vapour['pore.surface_volume_wp'][0]/p_volumes[0])

        if moist_volume:
            phys_vapour.models.add(propname='pore.moist_volume_wp',
                                   model=pm.volume_moisture.pvolume,
                                   pc=Pc_step_wp)
            phys_vapour.models.add(propname='throat.moist_volume_wp',
                                   model=pm.volume_moisture.tvolume,
                                   pc=Pc_step_wp)
            phys_vapour.models.add(propname='pore.moist_volume_drying',
                                   model=pm.volume_moisture.pvolume,
                                   pc=Pc_step_dp,
                                   pore_occupancy='pore.occupancy_drying')
            phys_vapour.models.add(propname='throat.moist_volume_drying',
                                   model=pm.volume_moisture.tvolume,
                                   pc=Pc_step_dp,
                                   throat_occupancy='throat.occupancy_drying')


            volume_p_wp = sum(phys_vapour['pore.moist_volume_wp'])
            volume_t_wp = sum(phys_vapour['throat.moist_volume_wp'])
            volume_p_dp = sum(phys_vapour['pore.moist_volume_drying'])
            volume_t_dp = sum(phys_vapour['throat.moist_volume_drying'])

            sat_moist_wp = (volume_p_wp + volume_t_wp)/(volume_total)
            sat_moist_dp = (volume_p_dp + volume_t_dp)/(volume_total)
            sat_wp_moist.append(sat_moist_wp)
            sat_dp_moist.append(sat_moist_dp)

            sat_wp[-1] += sat_moist_wp   # update total saturations
            sat_dp[-1] += sat_moist_dp

            print('moist vol: ', phys_vapour['throat.moist_volume_wp'])
            print('moist sat: ',
                  phys_vapour['throat.moist_volume_wp']/t_volumes)

        # Update vapour permeability for ALL pores & throats for wp & dp
        # g = f(Pv(rh(pc)))
        phys_vapour.models.add(propname='throat.diffusive_conductance_wp',
                               model=pm.diffusive_conductance.tbulk_diffusion,
                               pc=Pc_step_wp, knudsen=knudsen,
                               film_thickness='throat.surface_thickness_wp')
        phys_vapour.models.add(propname='throat.diffusive_conductance_wp_pore',
                                model=pm.diffusive_conductance.tbulk_diffusion_pore,
                                pc=Pc_step_wp, knudsen=knudsen,
                                film_thickness='pore.surface_thickness_wp')

        phys_vapour.models.add(propname='throat.diffusive_conductance_drying',
                               model=pm.diffusive_conductance.tbulk_diffusion,
                               pc=Pc_step_dp, knudsen=knudsen,
                               film_thickness='throat.surface_thickness_drying')
        phys_vapour.models.add(propname='throat.diffusive_conductance_drying_pore',
                                model=pm.diffusive_conductance.tbulk_diffusion_pore,
                                pc=Pc_step_dp, knudsen=knudsen,
                                film_thickness='pore.surface_thickness_drying')
        phys_vapour.models.regenerate()

        # Calculate conduit conductances as a function of water distribution
        phys_moisture.models.add(propname='throat.conduit_conductance_wp',
                                 model=pm.multiphase.mixed_conductance_pore,
                                 throat_occupancy='throat.'+occupy,
                                 pore_occupancy='pore.'+occupy,
                                 pdiffusive_conductance='throat.diffusive_conductance_wp_pore',
                                 tdiffusive_conductance='throat.diffusive_conductance_wp',
                                 diffusion=diffusion)
        phys_moisture.models.add(propname='throat.conduit_conductance_drying',
                                 model=pm.multiphase.mixed_conductance_pore,
                                 throat_occupancy='throat.occupancy_drying',
                                 pore_occupancy='pore.occupancy_drying',
                                 pdiffusive_conductance='throat.diffusive_conductance_drying_pore',
                                 tdiffusive_conductance='throat.diffusive_conductance_drying',
                                 diffusion=diffusion)

        phys_moisture.models.regenerate()

        bounds = [['inlet', 'outlet']]
        pc1_wp = Pc_step_wp + dPc
        pc2_wp = Pc_step_wp - dPc
        pc1_dp = Pc_step_dp + dPc
        pc2_dp = Pc_step_dp - dPc

        for bound_increment in range(len(bounds)):
            alg_flow_wp = pab.MoistureFlow(name='alg_flow_wp', network=pn,
                                           phase=moisture)
            alg_flow_dp = pab.MoistureFlow(name='alg_flow_dp', network=pn,
                                           phase=moisture)

            BC1_pores = pn.pores(labels=bounds[bound_increment][0])
            BC2_pores = pn.pores(labels=bounds[bound_increment][1])

            # BC1
            alg_flow_wp.set_boundary_conditions(bctype='Dirichlet',
                                                bcvalue=pc1_wp,
                                                pores=BC1_pores)
            alg_flow_dp.set_boundary_conditions(bctype='Dirichlet',
                                                bcvalue=pc1_dp,
                                                pores=BC1_pores)

            # BC2
            alg_flow_wp.set_boundary_conditions(bctype='Dirichlet',
                                                bcvalue=pc2_wp,
                                                pores=BC2_pores)
            alg_flow_dp.set_boundary_conditions(bctype='Dirichlet',
                                                bcvalue=pc2_dp,
                                                pores=BC2_pores)

            # run algorithms with proper conduit conductance
            alg_flow_wp.run(conductance='conduit_conductance_wp',
                            quantity='pressure_wp')
            alg_flow_dp.run(conductance='conduit_conductance_drying',
                                quantity='pressure_drying')

            # calc effective permeabilities [s]
            eff_permeability_moisture_wp = alg_flow_wp.calc_eff_permeability(
                conductance=phys_moisture['throat.conduit_conductance_wp'])
            eff_permeability_moisture_dp = alg_flow_dp.calc_eff_permeability(
                conductance=phys_moisture['throat.conduit_conductance_drying'])

            # append permeability & flow values to the lists
            eff_perm_moisture_wp[str(bound_increment)].append(
                eff_permeability_moisture_wp)
            eff_perm_moisture_dp[str(bound_increment)].append(
                eff_permeability_moisture_dp)

            ctrl.purge_object(alg_flow_wp)
            ctrl.purge_object(alg_flow_dp)

    alg_flow_wp.store_result(Pc=Pc_wp, sat=sat_wp, sat_surf=sat_wp_surf,
                             sat_moist=sat_wp_moist,
                             w_sat=w_sat, k=eff_perm_moisture_wp['0'])
    alg_flow_dp.store_result(Pc=Pc_wp, sat=sat_dp, sat_surf=sat_dp_surf,
                             sat_moist=sat_dp_moist,
                             w_sat=w_sat, k=eff_perm_moisture_dp['0'])

    alg_flow_wp.calc_abs_permeability()
    alg_flow_wp.calc_mD_permeability()
    alg_flow_dp.calc_abs_permeability()
    alg_flow_dp.calc_mD_permeability()

    if plot:
        bpnm.Postprocessing.Plots.hysteresis(alg_flow_wp, alg_flow_dp)
    return (alg_flow_wp, alg_flow_dp)


def permeability2(pn, alg_wp, alg_dp, water, vapour, moisture,
                 phys_vapour, phys_moisture, w_sat, cases=['wetting'],
                 trapping=False,
                 num_seq=10, knudsen=True, plot=True, printstatus=False,
                 surface_ad=False, moist_volume=False, dPc =1, diffusion=True,
                 refine=False, debug=False,
                 **kwargs):
    r"""
    Solver:
    -------
    direct solver
        amg=None (default), and iterative_solver=None (default)

    iterative solver
        amg=None, and iterative_solver = 'cg', 'gmres'

    amg iterative solver
        amg='rs', 'classic', 'sa', 'rootnode', 'blackbox'
    """


    pm = bpnm.Physics.models
    pab = bpnm.Algorithms

    for case in cases:
        pcs = []
        sat_wp, sat_dp = [], []
        sat_wp_surf, sat_dp_surf = [], []
        sat_wp_moist, sat_dp_moist = [], []
        eff_perm_moisture_wp = {'0': [], '1': [], '2': []}
        eff_perm_moisture_dp = {'0': [], '1': [], '2': []}

        p_volumes = pn['pore.volume']
        t_volumes = pn['throat.volume']
        volume_total = sum(p_volumes) + sum(t_volumes)

        lr = sp.arange(-10, -1)
        r = sp.power(10, lr)
        pc = -2*water['pore.surface_tension'][0]/r

        Pc_min, Pc_max = sp.amin(pc), sp.amax(pc)
        Pc_wp = -sp.logspace(sp.log10(-Pc_min), sp.log10(-Pc_max), num_seq)
        Pc_list = sp.copy(Pc_wp)
        Pc_wp = list(Pc_wp[::-1])

        occupancy = 'occupancy_' + case
        occupancy_dp = 'occupancy_drying_' + case
        case_dp = 'drying_' + case
        if trapping:
            occupy = occupancy + '_trapping'
        else:
            occupy = occupancy
        i = 0
        while len(Pc_wp) > 0:
            Pc_step_wp = Pc_wp.pop()
            Pc_step_dp = Pc_step_wp

            alg_wp.return_results(Pc=Pc_step_wp, cases=[case],
                                  occupancies=[occupancy], trapping=trapping)
            alg_dp.return_results(Pc=Pc_step_dp, cases=[case_dp],
                                  occupancies=[occupancy_dp],
                                  trapping=False)

            p_occ_wp = water['pore.'+occupy]
            t_occ_wp = water['throat.'+occupy]
            p_occ_dp = water['pore.'+occupancy_dp]
            t_occ_dp = water['throat.'+occupancy_dp]

            volume_p_wp = sum(p_occ_wp*p_volumes)
            volume_t_wp = sum(t_occ_wp*t_volumes)
            volume_p_dp = sum(p_occ_dp*p_volumes)
            volume_t_dp = sum(t_occ_dp*t_volumes)

            saturation_wp = (volume_p_wp + volume_t_wp)/(volume_total)
            saturation_dp = (volume_p_dp + volume_t_dp)/(volume_total)

            sat_wp.append(saturation_wp)
            sat_dp.append(saturation_dp)
            pcs.append(Pc_step_wp)

            # Surface adsorption: p/t.surface_thickness & surface_volume
            if surface_ad:
                phys_vapour.models.add(propname='pore.surface_thickness_'+case,
                                       model=pm.surface_adsorption.pstat_thickness,
                                       pc=Pc_step_wp,
                                       pore_occupancy='pore.'+occupy)
                phys_vapour.models.add(propname='throat.surface_thickness_'+case,
                                       model=pm.surface_adsorption.tstat_thickness,
                                       pc=Pc_step_wp,
                                       throat_occupancy='throat.'+occupy)
                phys_vapour.models.add(propname='pore.surface_thickness_'+case_dp,
                                       model=pm.surface_adsorption.pstat_thickness,
                                       pc=Pc_step_dp,
                                       pore_occupancy='pore.'+occupancy_dp)
                phys_vapour.models.add(propname='throat.surface_thickness_'+case_dp,
                                       model=pm.surface_adsorption.tstat_thickness,
                                       pc=Pc_step_dp,
                                       throat_occupancy='throat.'+occupancy_dp)

                ## Inner circular tube
                phys_vapour.models.add(propname='pore.surface_volume_'+case,
                                       model=pm.surface_adsorption.pvolume,
                                       pore_occupancy='pore.'+occupy,
                                       film_thickness='pore.surface_thickness_'+case)
                phys_vapour.models.add(propname='throat.surface_volume_'+case,
                                       model=pm.surface_adsorption.tvolume,
                                       throat_occupancy='throat.'+occupy,
                                       film_thickness='throat.surface_thickness_'+case)
                phys_vapour.models.add(propname='pore.surface_volume_'+case_dp,
                                       model=pm.surface_adsorption.pvolume,
                                       pore_occupancy='pore.'+occupancy_dp,
                                       film_thickness='pore.surface_thickness_'+case_dp)
                phys_vapour.models.add(propname='throat.surface_volume_'+case_dp,
                                       model=pm.surface_adsorption.tvolume,
                                       throat_occupancy='throat.'+occupancy_dp,
                                       film_thickness='throat.surface_thickness_'+case_dp)


                volume_p_wp = sum(phys_vapour['pore.surface_volume_'+case])
                volume_t_wp = sum(phys_vapour['throat.surface_volume_'+case])
                volume_p_dp = sum(phys_vapour['pore.surface_volume_'+case_dp])
                volume_t_dp = sum(phys_vapour['throat.surface_volume_'+case_dp])

                sat_surf_wp = (volume_p_wp + volume_t_wp)/(volume_total)
                sat_surf_dp = (volume_p_dp + volume_t_dp)/(volume_total)
                sat_wp_surf.append(sat_surf_wp)
                sat_dp_surf.append(sat_surf_dp)

                sat_wp[-1] += sat_surf_wp   # update total saturations
                sat_dp[-1] += sat_surf_dp

            # Update vapour permeability for ALL pores & throats for wp & dp
            # g = f(Pv(rh(pc)))
            # for the same pcwp & pcdp, diffusive_conductance of wp & dp are the same
            phys_vapour.models.add(propname='throat.diffusive_conductance_'+case,
                                   model=pm.diffusive_conductance.tbulk_diffusion,
                                   pc=Pc_step_wp, knudsen=knudsen,
                                   film_thickness='throat.surface_thickness_'+case)
            phys_vapour.models.add(propname='throat.diffusive_conductance_wp_pore',
                                    model=pm.diffusive_conductance.tbulk_diffusion_pore,
                                    pc=Pc_step_wp, knudsen=knudsen,
                                    film_thickness='pore.surface_thickness_'+case)

            phys_vapour.models.add(propname='throat.diffusive_conductance_'+case_dp,
                                   model=pm.diffusive_conductance.tbulk_diffusion,
                                   pc=Pc_step_dp, knudsen=knudsen,
                                   film_thickness='throat.surface_thickness_'+case_dp)
            phys_vapour.models.add(propname='throat.diffusive_conductance_drying_pore',
                                    model=pm.diffusive_conductance.tbulk_diffusion_pore,
                                    pc=Pc_step_dp, knudsen=knudsen,
                                    film_thickness='pore.surface_thickness_'+case_dp)
            phys_vapour.models.regenerate()

            # Calculate conduit conductances as a function of water distribution
            phys_moisture.models.add(propname='throat.conduit_conductance_'+case,
                                     model=pm.multiphase.mixed_conductance_pore,
                                     throat_occupancy='throat.'+occupy,
                                     pore_occupancy='pore.'+occupy,
                                     pdiffusive_conductance='throat.diffusive_conductance_wp_pore',
                                     tdiffusive_conductance='throat.diffusive_conductance_'+case,
                                     diffusion=diffusion)
            phys_moisture.models.add(propname='throat.conduit_conductance_'+case_dp,
                                     model=pm.multiphase.mixed_conductance_pore,
                                     throat_occupancy='throat.occupancy_'+case_dp,
                                     pore_occupancy='pore.occupancy_'+case_dp,
                                     pdiffusive_conductance='throat.diffusive_conductance_drying_pore',
                                     tdiffusive_conductance='throat.diffusive_conductance_'+case_dp,
                                     diffusion=diffusion)

            phys_moisture.models.regenerate()

            bounds = [['inlet', 'outlet']]
            pc1_wp = Pc_step_wp + dPc
            pc2_wp = Pc_step_wp - dPc
            pc1_dp = Pc_step_dp + dPc
            pc2_dp = Pc_step_dp - dPc

            kmin = phys_moisture['throat.conduit_conductance_'+case_dp].min()
            kmax = phys_moisture['throat.conduit_conductance_'+case_dp].max()
            print('Pc: {}'.format(Pc_step_wp))

            for bound_increment in range(len(bounds)):
                alg_flow_wp = pab.MoistureFlow(name='alg_flow_'+case, network=pn,
                                               phase=moisture)
                alg_flow_dp = pab.MoistureFlow(name='alg_flow_dp', network=pn,
                                               phase=moisture)

                BC1_pores = pn.pores(labels=bounds[bound_increment][0])
                BC2_pores = pn.pores(labels=bounds[bound_increment][1])

                # BC1
                alg_flow_wp.set_boundary_conditions(bctype='Dirichlet',
                                                    bcvalue=pc1_wp,
                                                    pores=BC1_pores)
                alg_flow_dp.set_boundary_conditions(bctype='Dirichlet',
                                                    bcvalue=pc1_dp,
                                                    pores=BC1_pores)

                # BC2
                alg_flow_wp.set_boundary_conditions(bctype='Dirichlet',
                                                    bcvalue=pc2_wp,
                                                    pores=BC2_pores)
                alg_flow_dp.set_boundary_conditions(bctype='Dirichlet',
                                                    bcvalue=pc2_dp,
                                                    pores=BC2_pores)

                # run algorithms with proper conduit conductance
                x0 = Pc_step_wp
#                if sp.absolute(saturation_wp*w_sat-39) < 2:
#                    pass
#                x0dp = sp.ones_like(b)*Pc_step_dp
                alg_flow_wp.run2(conductance='conduit_conductance_'+case,
                                quantity='pressure_wp', x0=x0, **kwargs)
                alg_flow_dp.run2(conductance='conduit_conductance_'+case_dp,
                                 quantity='pressure_drying', x0=x0, **kwargs)


#                folder = '/home/islah/Documents/python3/multiscale/regular_multiscale_net/'
#                if sp.absolute(saturation_wp*w_sat-0) == 0:
##                    pass
#                    bpnm.Utilities.IO.save_sparse_csr(folder+'A_multi_w0',
#                                                      alg_flow_wp.A)
#                    sp.save(folder+'b_multi_w0', alg_flow_wp.b)
#                if sp.absolute(saturation_wp*w_sat-1.7e-5) < 1e-5:
#                    bpnm.Utilities.IO.save_sparse_csr(folder+'A_multi_w1p7e5',
#                                                      alg_flow_wp.A)
#                    sp.save(folder+'b_multi_w1p7e5', alg_flow_wp.b)
#                if sp.absolute(saturation_wp*w_sat-3.9) < 0.2:
#                    bpnm.Utilities.IO.save_sparse_csr(folder+'A_berea_w3p9_wet',
#                                                      alg_flow_wp.A)
#                    sp.save(folder+'b_berea_w3p9', alg_flow_wp.b)
#                if sp.absolute(saturation_wp*w_sat-50) < 10:
#                    sp.save(folder+'b_berea_w50', alg_flow_wp.b)
#                if sp.absolute(saturation_wp*w_sat-140) < 10:
#                    sp.save(folder+'b_berea_w140', alg_flow_wp.b)
#                if sp.absolute(saturation_wp*w_sat-50) < 10:
#                    sp.save(folder+'b_berea_w50', alg_flow_wp.b)

                # calc effective permeabilities [s]
                eff_permeability_moisture_wp = alg_flow_wp.calc_eff_permeability(
                    conductance=phys_moisture['throat.conduit_conductance_'+case])
                eff_permeability_moisture_dp = alg_flow_dp.calc_eff_permeability(
                    conductance=phys_moisture['throat.conduit_conductance_'+case_dp])

                # append permeability & flow values to the lists
                eff_perm_moisture_wp[str(bound_increment)].append(
                    eff_permeability_moisture_wp)
                eff_perm_moisture_dp[str(bound_increment)].append(
                    eff_permeability_moisture_dp)

                alg_flow_wp.return_rate(case='_'+case)
                alg_flow_dp.return_rate(case='_'+case_dp)

                ctrl.purge_object(alg_flow_wp)
                ctrl.purge_object(alg_flow_dp)
            # Grid refinement
            if len(Pc_wp)==0 and refine:
                arg = sp.argsort(pcs)
                pcs = sp.array(pcs)[arg]
                sat_wp = sp.array(sat_wp)[arg]
                eff_perm_moisture_wp['0'] = sp.array(eff_perm_moisture_wp['0'])[arg]
                sat_dp = sp.array(sat_dp)[arg]
                eff_perm_moisture_dp['0'] = sp.array(eff_perm_moisture_dp['0'])[arg]
                Pc_list = Pc_list[arg]

                if surface_ad:
                    sat_wp_surf = sp.array(sat_wp_surf)[arg]
                    sat_dp_surf = sp.array(sat_dp_surf)[arg]

                dw = sp.absolute(sp.diff(sat_wp))
                dw2 = sp.absolute(sp.diff(sat_dp))
                tag = sp.where(dw>0.2)[0]
                tag2 = sp.where(dw2>0.2)[0]
                tag = sp.unique(sp.r_[tag, tag2])
                tagged = sp.array([Pc_list[tag], Pc_list[tag+1]])
                fine = -sp.power(10, sp.mean(sp.log10(-tagged), axis=0))
                if len(fine)>0:
                    Pc_wp.extend(fine)
                    Pc_list = sp.concatenate((Pc_list,fine[::-1]))

                    pcs = list(pcs)
                    sat_wp = list(sat_wp)
                    sat_wp_surf = list(sat_wp_surf)
                    eff_perm_moisture_wp['0'] = list(eff_perm_moisture_wp['0'])
                    sat_dp = list(sat_dp)
                    sat_dp_surf = list(sat_dp_surf)
                    eff_perm_moisture_dp['0'] = list(eff_perm_moisture_dp['0'])
            i += 1
            A = sp.absolute(alg_flow_dp.A.data)
            print('Amin: {}, Amax: {}'.format(A.min(), A.max()))

        arg = sp.argsort(pcs)
        pcwp = sp.array(pcs)[arg]
        sat_wp = sp.array(sat_wp)[arg]
        k = sp.array(eff_perm_moisture_wp['0'])[arg]
        if surface_ad:
            sat_wp_surf = sp.array(sat_wp_surf)[arg]
        alg_flow_wp.store_result(Pc=pcwp, sat=sat_wp,
                                 sat_surf=sat_wp_surf,
                                 sat_moist=sat_wp_moist,
                                 w_sat=w_sat, k=k)

        pcs = sp.array(pcs)[arg]
        sat_dp = sp.array(sat_dp)[arg]
        k = sp.array(eff_perm_moisture_dp['0'])[arg]
        if surface_ad:
            sat_dp_surf = sp.array(sat_dp_surf)[arg]
        alg_flow_dp.store_result(Pc=pcs, sat=sat_dp,
                                 sat_surf=sat_dp_surf,
                                 sat_moist=sat_dp_moist,
                                 w_sat=w_sat, k=k)

        alg_flow_wp.calc_abs_permeability()
        alg_flow_wp.calc_mD_permeability()
        alg_flow_dp.calc_abs_permeability()
        alg_flow_dp.calc_mD_permeability()

    if plot:
        bpnm.Postprocessing.Plots.hysteresis(alg_flow_wp, alg_flow_dp,
                                             legend=[case, 'drying'])
    return (alg_flow_wp, alg_flow_dp)

def permeability2_parallel(pn, alg_wp, alg_dp, water, vapour, moisture,
                 phys_vapour, phys_moisture, w_sat, cases=['wetting'],
                 trapping=False,
                 num_seq=10, knudsen=True, plot=True, printstatus=False,
                 surface_ad=False, moist_volume=False, dPc =1, diffusion=True,
                 refine=False, debug=False,
                 **kwargs):
    r"""
    Parallel version of permeability2(). However the method doesn't work due to
    a problem in updating phys_vapour for various pcs.
    """
    from multiprocessing import Pool, cpu_count

    pm = bpnm.Physics.models
    pab = bpnm.Algorithms

    for case in cases:
        pcs = []
        sat_wp, sat_dp = [], []
        sat_wp_surf, sat_dp_surf = [], []
        sat_wp_moist, sat_dp_moist = [], []
        eff_perm_moisture_wp = {'0': [], '1': [], '2': []}
        eff_perm_moisture_dp = {'0': [], '1': [], '2': []}

        p_volumes = pn['pore.volume']
        t_volumes = pn['throat.volume']
        volume_total = sum(p_volumes) + sum(t_volumes)

        lr = sp.arange(-10, -1)
        r = sp.power(10, lr)
        pc = -2*water['pore.surface_tension'][0]/r

        Pc_min, Pc_max = sp.amin(pc), sp.amax(pc)
        Pc_wp = -sp.logspace(sp.log10(-Pc_min), sp.log10(-Pc_max), num_seq)
        Pc_list = sp.copy(Pc_wp)
        Pc_wp = list(Pc_wp[::-1])

        occupancy = 'occupancy_' + case
        occupancy_dp = 'occupancy_drying_' + case
        case_dp = 'drying_' + case
        if trapping:
            occupy = occupancy + '_trapping'
        else:
            occupy = occupancy
        i = 0
#        while len(Pc_wp) > 0:
        argums = [(pc, i, occupy, occupancy, occupancy_dp, case_dp, Pc_list,
                   volume_total, pm, pab, case)\
                 for pc in Pc_wp[::-1]]
        pool = Pool(cpu_count()-1)
        pool.map(_calc_k_per_pcp, argums)
        pool.close()
        pool.join()

        arg = sp.argsort(pcs)
        pcwp = sp.array(pcs)[arg]
        sat_wp = sp.array(sat_wp)[arg]
        k = sp.array(eff_perm_moisture_wp['0'])[arg]
        if surface_ad:
            sat_wp_surf = sp.array(sat_wp_surf)[arg]
        alg_flow_wp.store_result(Pc=pcwp, sat=sat_wp,
                                 sat_surf=sat_wp_surf,
                                 sat_moist=sat_wp_moist,
                                 w_sat=w_sat, k=k)

        pcs = sp.array(pcs)[arg]
        sat_dp = sp.array(sat_dp)[arg]
        k = sp.array(eff_perm_moisture_dp['0'])[arg]
        if surface_ad:
            sat_dp_surf = sp.array(sat_dp_surf)[arg]
        alg_flow_dp.store_result(Pc=pcs, sat=sat_dp,
                                 sat_surf=sat_dp_surf,
                                 sat_moist=sat_dp_moist,
                                 w_sat=w_sat, k=k)

        alg_flow_wp.calc_abs_permeability()
        alg_flow_wp.calc_mD_permeability()
        alg_flow_dp.calc_abs_permeability()
        alg_flow_dp.calc_mD_permeability()

    if plot:
        bpnm.Postprocessing.Plots.hysteresis(alg_flow_wp, alg_flow_dp,
                                             legend=[case, 'drying'])
    return (alg_flow_wp, alg_flow_dp)


def _calc_k_per_pcp(args):
    (pc, i, occupy, occupancy, occupancy_dp, case_dp, Pc_list, volume_total,
     pm, pab, case, trapping, p_volumes, t_volumes, sat_wp, sat_dp, pcs,
     surface_ad, sat_wp_surf, sat_dp_surf, knudsen, diffusion, dPc,
     eff_perm_moisture_wp, eff_perm_moisture_dp, refine, Pc_wp, ) = args

    Pc_step_wp = pc
    Pc_step_dp = Pc_step_wp

    alg_wp.return_results(Pc=Pc_step_wp, cases=[case],
                          occupancies=[occupancy], trapping=trapping)
    alg_dp.return_results(Pc=Pc_step_dp, cases=[case_dp],
                          occupancies=[occupancy_dp],
                          trapping=False)

    p_occ_wp = water['pore.'+occupy]
    t_occ_wp = water['throat.'+occupy]
    p_occ_dp = water['pore.'+occupancy_dp]
    t_occ_dp = water['throat.'+occupancy_dp]

    volume_p_wp = sum(p_occ_wp*p_volumes)
    volume_t_wp = sum(t_occ_wp*t_volumes)
    volume_p_dp = sum(p_occ_dp*p_volumes)
    volume_t_dp = sum(t_occ_dp*t_volumes)

    saturation_wp = (volume_p_wp + volume_t_wp)/(volume_total)
    saturation_dp = (volume_p_dp + volume_t_dp)/(volume_total)

    sat_wp.append(saturation_wp)
    sat_dp.append(saturation_dp)
    pcs.append(Pc_step_wp)

    # Surface adsorption: p/t.surface_thickness & surface_volume
    if surface_ad:
        phys_vapour.models.add(propname='pore.surface_thickness_'+case,
                               model=pm.surface_adsorption.pstat_thickness,
                               pc=Pc_step_wp,
                               pore_occupancy='pore.'+occupy)
        phys_vapour.models.add(propname='throat.surface_thickness_'+case,
                               model=pm.surface_adsorption.tstat_thickness,
                               pc=Pc_step_wp,
                               throat_occupancy='throat.'+occupy)
        phys_vapour.models.add(propname='pore.surface_thickness_'+case_dp,
                               model=pm.surface_adsorption.pstat_thickness,
                               pc=Pc_step_dp,
                               pore_occupancy='pore.'+occupancy_dp)
        phys_vapour.models.add(propname='throat.surface_thickness_'+case_dp,
                               model=pm.surface_adsorption.tstat_thickness,
                               pc=Pc_step_dp,
                               throat_occupancy='throat.'+occupancy_dp)

        ## Inner circular tube
        phys_vapour.models.add(propname='pore.surface_volume_'+case,
                               model=pm.surface_adsorption.pvolume,
                               pore_occupancy='pore.'+occupy,
                               film_thickness='pore.surface_thickness_'+case)
        phys_vapour.models.add(propname='throat.surface_volume_'+case,
                               model=pm.surface_adsorption.tvolume,
                               throat_occupancy='throat.'+occupy,
                               film_thickness='throat.surface_thickness_'+case)
        phys_vapour.models.add(propname='pore.surface_volume_'+case_dp,
                               model=pm.surface_adsorption.pvolume,
                               pore_occupancy='pore.'+occupancy_dp,
                               film_thickness='pore.surface_thickness_'+case_dp)
        phys_vapour.models.add(propname='throat.surface_volume_'+case_dp,
                               model=pm.surface_adsorption.tvolume,
                               throat_occupancy='throat.'+occupancy_dp,
                               film_thickness='throat.surface_thickness_'+case_dp)


        volume_p_wp = sum(phys_vapour['pore.surface_volume_'+case])
        volume_t_wp = sum(phys_vapour['throat.surface_volume_'+case])
        volume_p_dp = sum(phys_vapour['pore.surface_volume_'+case_dp])
        volume_t_dp = sum(phys_vapour['throat.surface_volume_'+case_dp])

        sat_surf_wp = (volume_p_wp + volume_t_wp)/(volume_total)
        sat_surf_dp = (volume_p_dp + volume_t_dp)/(volume_total)
        sat_wp_surf.append(sat_surf_wp)
        sat_dp_surf.append(sat_surf_dp)

        sat_wp[-1] += sat_surf_wp   # update total saturations
        sat_dp[-1] += sat_surf_dp

    # Update vapour permeability for ALL pores & throats for wp & dp
    # g = f(Pv(rh(pc)))
    # for the same pcwp & pcdp, diffusive_conductance of wp & dp are the same
    phys_vapour.models.add(propname='throat.diffusive_conductance_'+case,
                           model=pm.diffusive_conductance.tbulk_diffusion,
                           pc=Pc_step_wp, knudsen=knudsen,
                           film_thickness='throat.surface_thickness_'+case)
    phys_vapour.models.add(propname='throat.diffusive_conductance_wp_pore',
                            model=pm.diffusive_conductance.tbulk_diffusion_pore,
                            pc=Pc_step_wp, knudsen=knudsen,
                            film_thickness='pore.surface_thickness_'+case)

    phys_vapour.models.add(propname='throat.diffusive_conductance_'+case_dp,
                           model=pm.diffusive_conductance.tbulk_diffusion,
                           pc=Pc_step_dp, knudsen=knudsen,
                           film_thickness='throat.surface_thickness_'+case_dp)
    phys_vapour.models.add(propname='throat.diffusive_conductance_drying_pore',
                            model=pm.diffusive_conductance.tbulk_diffusion_pore,
                            pc=Pc_step_dp, knudsen=knudsen,
                            film_thickness='pore.surface_thickness_'+case_dp)
    phys_vapour.models.regenerate()

    # Calculate conduit conductances as a function of water distribution
    phys_moisture.models.add(propname='throat.conduit_conductance_'+case,
                             model=pm.multiphase.mixed_conductance_pore,
                             throat_occupancy='throat.'+occupy,
                             pore_occupancy='pore.'+occupy,
                             pdiffusive_conductance='throat.diffusive_conductance_wp_pore',
                             tdiffusive_conductance='throat.diffusive_conductance_'+case,
                             diffusion=diffusion)
    phys_moisture.models.add(propname='throat.conduit_conductance_'+case_dp,
                             model=pm.multiphase.mixed_conductance_pore,
                             throat_occupancy='throat.occupancy_'+case_dp,
                             pore_occupancy='pore.occupancy_'+case_dp,
                             pdiffusive_conductance='throat.diffusive_conductance_drying_pore',
                             tdiffusive_conductance='throat.diffusive_conductance_'+case_dp,
                             diffusion=diffusion)

    phys_moisture.models.regenerate()

    bounds = [['inlet', 'outlet']]
    pc1_wp = Pc_step_wp + dPc
    pc2_wp = Pc_step_wp - dPc
    pc1_dp = Pc_step_dp + dPc
    pc2_dp = Pc_step_dp - dPc

#    kmin = phys_moisture['throat.conduit_conductance_'+case_dp].min()
#    kmax = phys_moisture['throat.conduit_conductance_'+case_dp].max()
    print('Pc: {}'.format(Pc_step_wp))

    for bound_increment in range(len(bounds)):
        alg_flow_wp = pab.MoistureFlow(name='alg_flow_'+case, network=pn,
                                       phase=moisture)
        alg_flow_dp = pab.MoistureFlow(name='alg_flow_dp', network=pn,
                                       phase=moisture)

        BC1_pores = pn.pores(labels=bounds[bound_increment][0])
        BC2_pores = pn.pores(labels=bounds[bound_increment][1])

        # BC1
        alg_flow_wp.set_boundary_conditions(bctype='Dirichlet',
                                            bcvalue=pc1_wp,
                                            pores=BC1_pores)
        alg_flow_dp.set_boundary_conditions(bctype='Dirichlet',
                                            bcvalue=pc1_dp,
                                            pores=BC1_pores)

        # BC2
        alg_flow_wp.set_boundary_conditions(bctype='Dirichlet',
                                            bcvalue=pc2_wp,
                                            pores=BC2_pores)
        alg_flow_dp.set_boundary_conditions(bctype='Dirichlet',
                                            bcvalue=pc2_dp,
                                            pores=BC2_pores)

        # run algorithms with proper conduit conductance
        x0 = Pc_step_wp

        alg_flow_wp.run2(conductance='conduit_conductance_'+case,
                        quantity='pressure_wp', x0=x0)
        alg_flow_dp.run2(conductance='conduit_conductance_'+case_dp,
                         quantity='pressure_drying', x0=x0)

        # calc effective permeabilities [s]
        eff_permeability_moisture_wp = alg_flow_wp.calc_eff_permeability(
            conductance=phys_moisture['throat.conduit_conductance_'+case])
        eff_permeability_moisture_dp = alg_flow_dp.calc_eff_permeability(
            conductance=phys_moisture['throat.conduit_conductance_'+case_dp])

        # append permeability & flow values to the lists
        eff_perm_moisture_wp[str(bound_increment)].append(
            eff_permeability_moisture_wp)
        eff_perm_moisture_dp[str(bound_increment)].append(
            eff_permeability_moisture_dp)

        alg_flow_wp.return_rate(case='_'+case)
        alg_flow_dp.return_rate(case='_'+case_dp)

        ctrl.purge_object(alg_flow_wp)
        ctrl.purge_object(alg_flow_dp)
    # Grid refinement
    if len(Pc_wp)==0 and refine:
        arg = sp.argsort(pcs)
        pcs = sp.array(pcs)[arg]
        sat_wp = sp.array(sat_wp)[arg]
        eff_perm_moisture_wp['0'] = sp.array(eff_perm_moisture_wp['0'])[arg]
        sat_dp = sp.array(sat_dp)[arg]
        eff_perm_moisture_dp['0'] = sp.array(eff_perm_moisture_dp['0'])[arg]
        Pc_list = Pc_list[arg]

        if surface_ad:
            sat_wp_surf = sp.array(sat_wp_surf)[arg]
            sat_dp_surf = sp.array(sat_dp_surf)[arg]

        dw = sp.absolute(sp.diff(sat_wp))
        dw2 = sp.absolute(sp.diff(sat_dp))
        tag = sp.where(dw>0.2)[0]
        tag2 = sp.where(dw2>0.2)[0]
        tag = sp.unique(sp.r_[tag, tag2])
        tagged = sp.array([Pc_list[tag], Pc_list[tag+1]])
        fine = -sp.power(10, sp.mean(sp.log10(-tagged), axis=0))
        if len(fine)>0:
            Pc_wp.extend(fine)
            Pc_list = sp.concatenate((Pc_list,fine[::-1]))

            pcs = list(pcs)
            sat_wp = list(sat_wp)
            sat_wp_surf = list(sat_wp_surf)
            eff_perm_moisture_wp['0'] = list(eff_perm_moisture_wp['0'])
            sat_dp = list(sat_dp)
            sat_dp_surf = list(sat_dp_surf)
            eff_perm_moisture_dp['0'] = list(eff_perm_moisture_dp['0'])
    i += 1
    A = sp.absolute(alg_flow_dp.A.data)
    print('Amin: {}, Amax: {}'.format(A.min(), A.max()))

def _check_occupancy(p_occ, t_occ, case, conns):
    if case=='wetting':
        pass
    elif case=='drying':
        t_pocc = p_occ[conns]
        psum = sp.sum(t_pocc, axis=0)

# %% A group of steps to calculate moisture retention curve
def moisture_storage(plot=True, **kwargs):
    pn, geoinput, geomacro = network(
        dat_file='test/smallBenth.p',
        trimming=True)
    geo = geometry(pn, geoinput=geoinput)
    water, vapour, moisture = phase(pn)
    phys_water, phys_vapour, phys_moisture = physic(pn, water, vapour,
                                                    moisture)
    alg_wp, alg_dp = algorithm(pn, water, vapour)
    v_mat = sp.prod(geomacro[b'size'])
    alg_wp, alg_dp, w_sat, porosity = moist_content(geo, alg_wp, alg_dp,
                                                    water['pore.density'][0],
                                                    v_mat=v_mat)
    if plot:
        plot_moist_retention(alg_wp)

    return(pn, geo, water, vapour, moisture, phys_water, phys_vapour,
           phys_moisture, alg_wp, alg_dp, w_sat, porosity)


# %% Main
if __name__ == '__main__':
    import operator as op
    from bwfpnm import routine_pore_percolation_new as bwfr

    #%% Load simulation
    filename = '/home/islah/Documents/01_Year-1/10_Papers/01_Static_single_scale/data/'
    filename += 'berea_wetting_drying.pnm'
    ctrl.load(filename)
    keys = ['net', 'geo', 'water', 'vapour', 'moisture',
            'physics_water', 'physics_vapour', 'physics_moisture',
            'percolation', 'percolation_dp']
    (pn, geo, water, vapour, moisture, phys_water, phys_vapour, phys_moisture,
     alg_wp, alg_dp) = op.itemgetter(*keys)(ctrl)

#==============================================================================
#     loc = '/home/islah/Documents/01_Year-1/10_Papers/01_Static_single_scale/data/'
#     #%% Load data
#     subdir = 'berea_Feb22/'
#     file_data = loc+ subdir + 'berea.p'
#     netinput, geoinput, geomacro = bpnm.Utilities.IO.load_data(file_data)
#     L = geomacro['size']
#     netinput.update(macro_Lx=L[0], macro_Ly=L[1], macro_Lz=L[2])
#     # %% Moisture content
#     pn = bwfr.network(bpnm.Network.Topology, netname='net', **netinput)
#
#     geo = bwfr.geometry(pn, bpnm.Geometry.Topology, geoname='geo',
#                         **geoinput)
#
#     water, vapour, moisture = bwfr.phase(pn)
#
#     PhysWaterClass = bpnm.Physics.Standard_Topology_pore
#     phys_water, phys_vapour, phys_moisture = bwfr.physic(
#         pn, water, vapour, moisture, PhysWaterClass=PhysWaterClass)
#
#     # Trimm all isolated clusters (has no inlet nor outlet pores)
#     span, dead, isolated = pn.cluster_types()
#     trim_pores = sp.hstack(isolated)
#     pn.trim(pores=trim_pores)   # geometry, phase, and physics are adjusted too!
#     pn['pore.index'] = sp.arange(0, pn.Np)
#     pn['throat.index'] = sp.arange(0, pn.Nt)
#
#     pc_list = sp.r_[water['pore.capillary_pressure'],
#                     water['throat.capillary_pressure']]
#     pc_list.sort()
#     #%% Algorithm: Imbibition
#     poutlet = pn['pore.outlet']+pn['pore.inlet']
#     pinlet = pn['pore.inlet']
#
#     case = 'wetting'
#     amg = None
#     inv_points = -sp.logspace(9, 2, 50)
#     inv_points = None
#     alg_wp, alg_dp = bwfr.algorithm(pn, water, vapour, cases=[case],
#                                     trapping=False, dp=True, inlet_imb=pinlet,
#                                     inlet_dry=pinlet,
#                                     outlet=poutlet, inv_points=inv_points)
#==============================================================================
    #%% Algorithm
    case = 'wetting'
    amg = None
    alg_wp, alg_dp, w_sat, porosity = bwfr.moist_content(geo, alg_wp,
                                                     water['pore.density'][0],
                                                     alg_dp=alg_dp,
                                                     cases=[case],
                                                     trapping=False)
    alg_flow_wp, alg_flow_dp = bwfr.permeability2(pn, alg_wp, alg_dp,
                                             water, vapour, moisture,
                                             phys_vapour, phys_moisture,
                                             w_sat=w_sat,
                                             cases=[case], trapping=False,
                                             num_seq=50,
                                             printstatus=False,
                                             surface_ad=True,
                                             moist_volume=False,
                                             knudsen=True, amg=amg, tol=1e-14,
                                             strength=('symmetric', {'theta': 0.01}),
                                             CF='CLJPc',
                                             agg='naive', smooth='energy')

#    alg_flow_wp, alg_flow_dp = permeability2_parallel(pn, alg_wp, alg_dp,
#                                                      water, vapour, moisture,
#                                                      phys_vapour, phys_moisture,
#                                                      w_sat, cases=[case],
#                                                      trapping=False, num_seq=50,
#                                                      knudsen=True, plot=True,
#                                                      printstatus=False,
#                                                      surface_ad=False,
#                                                      moist_volume=False, dPc =1,
#                                                      diffusion=True, refine=True,
#                                                      debug=False)
