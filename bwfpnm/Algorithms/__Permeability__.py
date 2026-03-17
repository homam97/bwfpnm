# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:56:27 2016

@author: islah
"""
import scipy as _sp
import bwfpnm as bpnm
import bwfpnm.Physics.models as pm
from bwfpnm.Algorithms import MoistureFlow, WaterFlow
from OpenPNM.Base import logging
from OpenPNM.Base import Core
import copy
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import string
import random

ctrl = bpnm.Base.Controller()
ctrl.loglevel = 40
logger = logging.getLogger(__name__)

Results = {}
DEBUG = False

class Permeability(Core):
    r"""
    A class for permeability computation.
    """
    def __init__(self, network, alg_wp,
                 phys_vapour, phys_moisture, alg_dp=None, **kwargs):
        super().__init__(**kwargs)
        logger.name = self.name

        self._pcs = {}
        self._sat = {}
        self._moisture_content = {}
#        self._sat_surf = {}
#        self._sat_moist = {}
        self._eff_permeability = {}
        self._eff_permeability2 = {}
        self._eff_permeability_vap = {}
        self._eff_permeability_vap2 = {}
        self._alg_flow = {}
        self._results = {}
        self._modified = False

        self._pvol = network['pore.volume']
        self._tvol = network['throat.volume']
        self._vol_total = _sp.sum(self._pvol) + _sp.sum(self._tvol)

        self._net = network
        self._alg_wp = alg_wp
        self._alg_dp = alg_dp
        self._phys_vapour = phys_vapour
        self._phys_moisture = phys_moisture

        try:
            span = network.pores('span')
        except:
            network.cluster_types(save=True)
            span = network.pores('span')

        if span.size > 0:
            self._is_span = True
        else:
            self._is_span = False

        for phase in network._phases:
            names = phase.name.split('_')
            if 'water' in names:
                self._water = phase
                self._phys_water = phase._physics[0]
            elif 'moisture' in names:
                self._moisture = phase
            elif 'vapour' in names:
                self._vapour = phase


    def _setup(self, knudsen, surface_ad, diffusion, refine, plot, printstatus,
               dPc, keep_log, debug, moist_vol, w_sat,
               single_flow, corner_ad, surf_flow,
               corner_flow, dsat_threshold=0.2,
               **kwargs):
        self._knudsen = knudsen
        self._surface_ad = surface_ad
        self._moist_vol = moist_vol
        self._diffusion = diffusion
        self._single_flow = single_flow
        self._corner_ad = corner_ad
        self._surf_flow = surf_flow
        self._corner_flow = corner_flow
        self._refine = refine
        self._plot = plot
        self._printstatus = printstatus
        self._dPc = dPc
        self._keep_log = keep_log
        self._debug = debug
        self._w_sat = w_sat
        self._adaptive_solver = False
#        try:
        self._dsat_threshold = dsat_threshold
#        except:
#            self._dsat_threshold = 0.2

        if kwargs['amg'] == 'adaptive':
            self._adaptive_solver = True
            try:
                self._thresholds = kwargs['wet_ratios']
                self._amg_solvers = kwargs['amg_solvers']
                self._tolerances = kwargs['tolerances']
                self._strengths = kwargs['strengths']
                self._CFs = kwargs['CFs']
                self._aggs = kwargs['aggs']
                self._smooths = kwargs['smooths']
            except:
                # klasses: hygroscopic, early/main/last transitions, nearly saturated,
                #           saturated.
                # self._amg_solvers[2] = 'ra' or None
                self._thresholds = [0.09, 0.35, 0.7, 0.95, 0.9995]
                self._amg_solvers = {0: 'rs', 1: 'rs', 2: None, 3: 'rs',
                                     4: 'rs', 5: 'ra'}
                self._tolerances = {0: 1e-13, 1: 1e-13, 2: 1e-14, 3: 1e-10,
                                    4: 1e-10, 5: 1e-10}
                self._strengths = {0: ('symmetric', {'theta': 0.03}),
                                   1: ('symmetric', {'theta': 0.01}),
                                   2: ('symmetric', {'theta': 0.01}),
                                   3: ('symmetric', {'theta': 0.01}),
                                   4: ('symmetric', {'theta': 0.03}),
                                   5: ('symmetric', {'theta': 0.01})}
                self._CFs = {i: 'RS' for i in range(6)}
                self._aggs = {i: 'standard' for i in range(6)}
                self._smooths = {0: 'gauss-seidel', 1: 'gauss-seidel',
                                 2: 'energy', 3: 'gauss-seidel',
                                 4: 'gauss-seidel', 5: 'energy'}
            try:
                if kwargs['accel'] is not None:
                    self._tolerances = {0: 1e-6, 1: 1e-8, 2: 1e-8, 3: 1e-6,
                                        4: 1e-6, 5: 1e-6}
                    self._amg_solvers[2] = 'rs'
                    self._amg_solvers[5] = 'rs'
            except:
                pass


    def _calc_pc_grids(self, num_seq=50, pc_grids=None, **kwargs):
        r"""
        Return:
        -------
        A list of pc grids of num_seq size.
        """
        if pc_grids is None:
            lr = _sp.arange(-10, -1)
            r = _sp.power(10, lr)
            pc = -2*self._water['pore.surface_tension'][0]/r

            pc_grids = -_sp.logspace(_sp.log10(-pc.min()), _sp.log10(-pc.max()),
                                  num_seq)

        else:
            pc_grids = _sp.array(pc_grids, ndmin=1)

        self._pc_grids = pc_grids[::-1]     # inverted coz later list.pop()


    def _define_cases(self, cases=['wetting', 'imbibition', 'drying'],
                      trapping=False):
        self._occupancy = {}
        self._occupy = {}
        self._cases = []
        self._trapping = trapping
        for case in cases:
            if case in ['wetting', 'imbibition']:
                occupancy = 'occupancy_' + case
                occupy = occupancy
                if trapping:
                    occupy = occupancy + '_trapping'

                self._occupancy[case] = occupancy
                self._occupy[case] = occupy
                self._cases.append(case)
#            elif case == 'drying_wetting':
#                self._occupancy['drying'] = {}
#                self._occupancy['drying']['wetting'] = 'occupancy_' + case
#                self._cases.append(case)
#            elif case == 'drying_imbibition':
#                self._occupancy['drying'] = {}
#                self._occupancy['drying']['wetting'] = 'occupancy_' + case
#                self._cases.append(case)
            elif case == 'drying':
                self._occupancy[case] = {}
                if 'wetting' in self._occupancy.keys():
                    case_dp = 'drying_wetting'
                    occupancy = 'occupancy_' + case_dp
                    self._occupancy[case]['wetting'] = occupancy
                    self._cases.append(case_dp)
                if 'imbibition' in self._occupancy.keys():
                    case_dp = 'drying_imbibition'
                    occupancy = 'occupancy_' + case_dp
                    self._occupancy[case]['imbibition'] = occupancy
                    self._cases.append(case_dp)
            else:
                raise Exception('Error: the case is not recognised.')


    def permeability_curve(self, cases=['wetting'], w_sat=1,
                           trapping=False, num_seq=10,
                           knudsen=True, plot=True, printstatus=False,
                           surface_ad=False, moist_volume=False, dPc =1,
                           diffusion=True, refine=False, keep_log=False,
                           moist_vol=False, debug=False,
                           single_flow=False, corner_ad=False, surf_flow=False,
                           corner_flow=False, par_pc=False, **kwargs):
        r"""
        case : 'wetting', 'imbibition', 'drying'
        note: for the invasion algorithm, the case argument for the drying is
            'drying' as long as the alg_dp is not None,
            here the correponding argument is ['wetting', 'drying'].

        the alg_flow result is stored in self._alg_flow
        """
        self._setup(knudsen, surface_ad, diffusion, refine, plot, printstatus,
                    dPc, keep_log, debug, moist_vol, w_sat,
                    single_flow, corner_ad, surf_flow, corner_flow, **kwargs)
        self._calc_pc_grids(num_seq, **kwargs)
        self._define_cases(cases, trapping)
        if surface_ad:
            self._sat_surf = {}
        if moist_vol:
            self._sat_moist = {}
        if corner_ad:
            self._sat_corner = {}

        if debug:
            self._pocc_number = {}
            self._tocc_number = {}
            self._res = {}
            self._span_number = {}
            self._dead_number = {}
            self._isolated_number = {}
            self._max_cluster_size = {}
            self._cwater_min, self._cwater_max = {}, {}
            self._cvapour_min, self._cvapour_max = {}, {}
            self._BCplane, self._flow2, self._flow = {}, {}, {}
            self._rel_residues = {}
#            self._tols = {}
            self._norm_b = {}
        if keep_log:
            self._time_solver = {}
            self._time_solver_wall = {}
            self._solver_list = {}
            self._N_iters = {}
#            self._rel_residues = {}
            self._tols = {}

        for case in self._cases:
            if case in ['wetting', 'imbibition']:
                occupancy = self._occupancy[case]
                occupy = self._occupy[case]
                alg = self._alg_wp
            elif case == 'drying_wetting':
                occupancy = self._occupancy['drying']['wetting']
                occupy = occupancy
                alg = self._alg_dp
            elif case == 'drying_imbibition':
                occupancy = self._occupancy['drying']['imbibition']
                occupy = occupancy
                alg = self._alg_dp

            if debug:
                self._pocc_number[case] = []
                self._tocc_number[case] = []
                self._res[case] = []
                self._span_number[case] = []
                self._dead_number[case] = []
                self._isolated_number[case] = []
                self._max_cluster_size[case] = []
                self._cwater_min[case], self._cwater_max[case] = [], []
                self._cvapour_min[case], self._cvapour_max[case] = [], []
                self._BCplane[case], self._flow2[case] = [], []
                self._flow[case] = []
                self._rel_residues[case] = []
#                self._tols[case] = []
                self._norm_b[case] = []
            if keep_log:
                self._time_solver[case] = []
                self._time_solver_wall[case] = []
                self._solver_list[case] = []
                self._N_iters[case] = []
#                self._rel_residues[case] = []
                self._tols[case] = []

            self._sat[case] = []
            self._pcs[case] = []
            self._eff_permeability[case] = {'moisture': [], 'water': [],
                                            'vapour': []}
            self._eff_permeability2[case] = {'moisture': [], 'water': [],
                                             'vapour': []}
            self._eff_permeability_vap[case] = {'moisture': [], 'water': [],
                                            'vapour': []}
            self._eff_permeability_vap2[case] = {'moisture': [], 'water': [],
                                             'vapour': []}
            Results.update({case:{'saturation': [], 'permeability': {},
                                  'permeability_vap': {}}})
            Results[case]['permeability'].update({'water': [],
                                                        'vapour': [],
                                                        'moisture': []})
            Results[case]['permeability_vap'].update({'water': [],
                                                        'vapour': [],
                                                        'moisture': []})
            if surface_ad:
                self._sat_surf[case] = []
                Results[case].update({'sat_surf': []})
            if moist_vol:
                self._sat_moist[case] = []
                Results[case].update({'sat_moist': []})
            if corner_ad:
                self._sat_corner[case] = []
                Results[case].update({'sat_corner': []})
            self._i = 0
            self._refine_iter = 0
            pc_list = list(_sp.sort(self._pc_grids)[::-1])
#            case_split = case.split('_')
            case_split = [case]
            
            if par_pc:
                result = self._par_permeability_curve(alg, case, case_split, occupancy,
                                             trapping, occupy, pc_list,
                                             surface_ad, corner_ad, par_pc,
                                             **kwargs)
                self._sorting_par(result, case)
            else:
                while len(pc_list) > 0:
                    pc = pc_list.pop()
                    # print('Case: {}, Pc: {}'.format(case, pc))
                    pc_list = self._permeability_curve(pc, alg, case, case_split, occupancy,
                                             trapping, occupy, pc_list,
                                             surface_ad, corner_ad, par_pc,
                                             **kwargs)
                    
                algs = _sp.array(self._alg_flow[case], ndmin=1)
                try:    # Works for permeable networks only
                    for alg in algs:
                        alg.calc_abs_permeability()
                        alg.calc_mD_permeability()
                except:
                    pass
                self._sorting(case)

            try: # remove comp error (sat > ~1)
                temp = _sp.array(self._sat[case])
                temp[temp>1] = 1.0
                self._sat[case] = list(temp)
            except:
                print('ERROR')
            self._moisture_content[case] = _sp.array(self._sat[case])*w_sat
            self._store_result(case)
        if plot:
            self.plot()


    def _par_permeability_curve(self, alg, case, case_split, occupancy,
                                trapping, occupy, pc_list, surface_ad,
                                corner_ad, par_pc, **kwargs):
        '''
        Compute the permeability in various Pcs at the same time

        Note
        - adapt the adaptive pc
        - passing kwargs to map?
        '''
        self._par_pc = par_pc
        kwargs.update({'alg0':alg, 'case':case, 'case_split':case_split,
                       'occupancy':occupancy, 'trapping':trapping,
                       'occupy':occupy, 'pc_list':pc_list,
                       'surface_ad':surface_ad, 'corner_ad':corner_ad,
                       'par_pc':par_pc})
        inputs = []
        while len(pc_list) > 0:
            pc = pc_list.pop()
            inputs.append(pc)

        pool = Pool(cpu_count()-1)

#        t = time.time()
        result = pool.map(partial(self._permeability_curve, **kwargs), inputs)
        pool.close()
        pool.join()
#        time_parallel = time.time() - t
        return result

    def _set_a_name(self, name=''):
        name += '_'+''.join(random.choice(string.ascii_uppercase +
                                         string.ascii_lowercase +
                                         string.digits) for _ in range(3))
        return name

#    @profile
    def _copy_alg(self, alg):
        alg_new = copy.copy(alg)
        alg_new._phases = alg_new._net._phases
        end = self._set_a_name()

        net = alg_new._net
        water = alg_new._phase_wet
        vapour = alg_new._phase_dry
        moisture = alg_new._phases[2]

        obs = [net, net._geometries[0], water, vapour, moisture,
               water._physics[0], vapour._physics[0], moisture._physics[0],
               alg_new]
        names = [ob.name+end for ob in obs]
        adict = dict(zip(names, obs))
        # update the name property of the new objects
        for name, ob in adict.items():
            ob.name = name
        # register the new objects to the controller
        ctrl.update(adict)
        return alg_new

#    @profile
    def _permeability_curve(self, pc, alg0, case, case_split, occupancy,
                            trapping,
                            occupy, pc_list, surface_ad, corner_ad,
                            par_pc=False, **kwargs):
        '''Problems:
            - the moisture distribution instances are mapped to
              the same phases objects => the whole simulation must be copied
              => memory demanding
        '''
        if par_pc:
            # Still doesn't work
            alg = self._copy_alg(alg0)
            print('Starting calculation for Pc = {}'.format(pc))

        else:
            alg = alg0
#            alg = self._copy_alg(alg0)

        alg.return_results(Pc=pc, cases=case_split,
                           occupancies=[occupancy], trapping=trapping)

        if not par_pc:
            self._alg_flow[case] = self._calc_each_pc(alg, pc, case, occupy,
                          case_split, occupancy, trapping, **kwargs)
        else:
            self._alg_flow, result = self._calc_each_pc(alg, pc, case, occupy,
                                                        case_split, occupancy,
                                                        trapping,**kwargs)
        if not par_pc:
            if self._sat[case][-1] >= 0.99 and pc_list:
                if _sp.diff(self._sat[case][-2:]) < 1e-6:
                    # if saturated
#                    self._sat[case].append(self._sat[case][-1])
#                    self._pcs[case].append(pc_list.pop())
                    keff = self._eff_permeability[case]
                    keff2 = self._eff_permeability2[case]
                    kveff = self._eff_permeability_vap[case]
                    kveff2 = self._eff_permeability_vap2[case]
                    self._append_last_values(case, pc_list)
                    for key in keff.keys():
                        try:    # without single flow, key==water&vapour will be failed
                            keff[key].append(keff[key][-1])
                            keff2[key].append(keff2[key][-1])
                            kveff[key].append(kveff[key][-1])
                            kveff2[key].append(kveff2[key][-1])
                        except:
                            pass
#                    if surface_ad:
#                        self._sat_surf[case].append(self._sat_surf[case][-1])
#                    if corner_ad:
#                        self._sat_corner[case].append(self._sat_corner[case][-1])
                    if self._refine and (self._refine_iter < 3):
                        pc_list = self._grid_refinement(case, pc_list,
                                                        self._dsat_threshold)
                        self._refine_iter += 1
                    else:
                        pc_list = []

            if len(pc_list)==0 and self._refine and (self._refine_iter < 3):
                # grid refinement
                pc_list = self._grid_refinement(case, pc_list, self._dsat_threshold)
                self._refine_iter += 1
            self._i += 1
            return pc_list
#            ctrl.purge_object(alg)
#            del alg
        else:
            sat, sat_surf, sat_corner, K1, K2 = result
            print('Ending calculation for Pc = {}'.format(pc))
            ctrl.purge_object(alg)
            del alg
            return (pc, sat, sat_surf, sat_corner, K1, K2)

    def _append_last_values(self, case, pc_list):
        self._sat[case].append(self._sat[case][-1])
        self._pcs[case].append(pc_list.pop())
        if self._surface_ad:
            self._sat_surf[case].append(self._sat_surf[case][-1])
        if self._corner_ad:
            self._sat_corner[case].append(self._sat_corner[case][-1])
        if self._debug:
            self._pocc_number[case].append(self._pocc_number[case][-1])
            self._tocc_number[case].append(self._tocc_number[case][-1])
            self._res[case].append(self._res[case][-1])
            self._span_number[case].append(self._span_number[case][-1])
            self._dead_number[case].append(self._dead_number[case][-1])
            self._isolated_number[case].append(self._isolated_number[case][-1])
            self._max_cluster_size[case].append(self._max_cluster_size[case][-1])
            self._cwater_min[case].append(self._cwater_min[case][-1])
            self._cwater_max[case].append(self._cwater_max[case][-1])
            self._cvapour_min[case].append(self._cvapour_min[case][-1])
            self._cvapour_max[case].append(self._cvapour_max[case][-1])
            self._BCplane[case].append(self._BCplane[case][-1])
            self._flow2[case].append(self._flow2[case][-1])
            self._flow[case].append(self._flow[case][-1])
#            self._time_solver[case] = []
#            self._solver_list[case] = []
#            self._N_iters[case] = []
            self._rel_residues[case].append(self._rel_residues[case][-1])
#            self._tols[case] = []
            self._norm_b[case].append(self._norm_b[case][-1])
        if self._keep_log:
            self._time_solver[case].append(self._time_solver[case][-1])
            self._time_solver_wall[case].append(self._time_solver_wall[case][-1])
            self._solver_list[case].append(self._solver_list[case][-1])
            self._N_iters[case].append(self._N_iters[case][-1])
#            self._rel_residues[case] = []
            self._tols[case].append(self._tols[case][-1])


    def _calc_each_pc(self, alg, pc, case, occupy,
                      case_split, occupancy, trapping, **kwargs):
        water = alg._phase_wet
        moisture = alg._net._phases[2]
        phys_water = water._physics[0]
        phys_vapour = alg._phase_dry._physics[0]
        phys_moisture = moisture._physics[0]
        
        p_occ = water['pore.'+occupy]
        t_occ = water['throat.'+occupy] 
        # print("p occ", p_occ)

#        if p_occ.sum()>0 or t_occ.sum()>0:
#            print('number of water filled elements: {}'.format(p_occ.sum()+t_occ.sum()))

        if self._debug:
            self._pocc_number[case].append(p_occ.sum())
            self._tocc_number[case].append(t_occ.sum())
            span, dead, isolated = self._net.cluster_types(mask=t_occ.astype('bool'), save=False)
            self._span_number[case].append(_sp.size(span))
            self._dead_number[case].append(_sp.size(dead))
            self._isolated_number[case].append(_sp.size(isolated))
            temp = self._max_cluster(span, dead, isolated)
            self._max_cluster_size[case].append(temp)
#            if span.size != 0:
#                span.size

        # === SATURATION ===
        volume_p = sum(p_occ*self._pvol)
        volume_t = sum(t_occ*self._tvol)
#        if pc

        saturation = (volume_p + volume_t)/(self._vol_total)

        self._sat[case].append(saturation)
        self._pcs[case].append(pc)
        Results[case]['saturation'].append([pc, saturation])

        if self._surface_ad:
            sat_surf = self._calc_surf_ad(pc, phys_water, phys_vapour, case,
                                          occupy)

        if self._corner_ad:
            sat_corner = self._calc_corner_ad(pc, phys_water, case, occupy)

            if 'imbibition' in case.split('_'):
                self._adjust_occupancy(water, case, occupy)

        if self._is_span:   # if the pores are spanning from inlet to outlet
            # === CONDUCTANCES ===
            self._vapour_conductance(pc, phys_vapour, case)
            if self._surf_flow or self._corner_flow:
                self._surface_conductance(pc, phys_water, case, occupy)

            self._conduit_conductance(pc, phys_moisture, case, occupy, **kwargs)

            if self._single_flow:
                self._conduit_single_conductance(pc, phys_water, phys_vapour,
                                                 case, occupy)

        # === PERMEABILITY ===
        bounds = [['inlet', 'outlet']]
        pc1 = pc + self._dPc
        pc2 = pc - self._dPc

        # --- Check occupancy consistency for pc1 - pc2 ---
        ppc = water['pore.capillary_pressure']
        tpc = water['throat.capillary_pressure']
        errmsg = "The dPc changes the moisture profile. Try reducing the dPc!\n"
        for item in [pc1, pc2]:
            alg.return_results(Pc=item, cases=case_split,
                           occupancies=[occupancy], trapping=trapping)
            p_occ1 = water['pore.'+occupy] #self._water['pore.'+occupy]
            t_occ1 = water['throat.'+occupy]
            dp_occ = p_occ == p_occ1
            dt_occ = t_occ == t_occ1
            pcond = ~_sp.all(dp_occ)
            tcond = ~_sp.all(dt_occ)
            if pcond or tcond:
                dpc = _sp.r_[ppc[~_sp.bool8(dp_occ)], tpc[~_sp.bool8(dt_occ)]]
                errmsg += 'pcbc, pmin, pmax: {}, {}, {}\n'.format(item,
                                             dpc.min(), dpc.max())
        try:
            dpc
            print(errmsg)
        except:
            pass
        alg.return_results(Pc=pc, cases=case_split,
                           occupancies=[occupancy], trapping=trapping)
#            assert _sp.all(p_occ == p_occ1), errmsg
#            assert _sp.all(t_occ == t_occ1), errmsg
        if self._adaptive_solver:
            kwargs = self._modify_solver(p_occ, t_occ, **kwargs)

        for j, bound in enumerate(bounds):
            alg_flow = self._calc_flow(moisture, pc, case, bound,
                                       pc1, pc2, 'moisture', p_occ=p_occ,
                                       t_occ=t_occ, **kwargs)

            if self._single_flow:
                vapour = alg._net._phases[1]
                water_flow = self._calc_flow(water, pc, case, bound,
                                             pc1, pc2, 'water',
                                             p_occ=p_occ, t_occ=t_occ,
                                             name='alg_flow_wtr',
                                             **kwargs)
                vapour_flow = self._calc_flow(vapour, pc, case, bound,
                                              pc1, pc2, 'vapour',
                                              p_occ=p_occ, t_occ=t_occ,
                                              name='alg_flow_vpr',
                                              **kwargs)

#        if self._debug:
#        A = _sp.absolute(alg_flow.A.data)
#        print('Amin: {}, Amax: {}'.format(A.min(), A.max()))
#        if self._single_flow:
#            return alg_flow, water_flow, vapour_flow
        try:
            self._par_pc
            alg_flow, K = alg_flow
            K1, K2 = K
            return (alg_flow, (saturation, sat_surf, sat_corner, K1, K2))
        except:
            return alg_flow


    def _calc_surf_ad(self, pc, phys_water, phys_vapour, case, occupy,
                      **kwargs):
        r"""
        Calculate surface thickness and volume due to surface adsorption for
        a given pc value.
        """
        vpr = phys_water
        element = ['pore', 'throat']
        prop = '.film_thickness_'+case
        for item in element:
            trap_pc = '.'.join([item, case+'_trapped_pc'])
            vpr.models.add(propname=item+prop,
                           model=pm.surface_adsorption.stat_thickness,
                           pc=pc, pore_occupancy=item+'.'+occupy,
                           Rv=phys_vapour._phases[0]['prop.gas_constant'],
                           trapping=self._trapping,
                           trap_pc=trap_pc,
                           film_thickness=item+prop)
#        vpr.models.regenerate()
#        if _sp.any(vpr['pore'+prop]):
#            temp = _sp.where(vpr['pore'+prop]>0)[0]

        prop1 = '.film_area_'+case
        for item in element:
            vpr.models.add(propname=item+prop1,
                           model=pm.surface_adsorption.area,
                           pore_occupancy=item+'.'+occupy,
                           film_thickness=item+prop)
#        vpr.models.regenerate()
#         # debug surface adsorption
#        temp = _sp.argsort(vpr._net['throat.shapefactor'])[0]
#        print('pc \t {}, t_film \t {}'.format(pc, vpr['throat'+prop][temp]))
#        print('\t A_film \t {}'.format(vpr['throat'+prop1][temp]))

        vol = 0
        prop2 = '.film_volume_'+case
        for item in element:
            vpr.models.add(propname=item+prop2,
                           model=pm.surface_adsorption.volume,
                           pore_occupancy=item+'.'+occupy,
                           film_area=item+prop1)
#            vpr.models.regenerate()
            vol += sum(vpr[item+prop2])
#        vpr.models.regenerate()

        sat_surf = vol/(self._vol_total)
        self._sat_surf[case].append(sat_surf)
        self._sat[case][-1] += sat_surf   # update total saturations
#        Results[case]['sat_surf'].append([pc, sat_surf])
#        Results[case]['saturation'][-1][1] += sat_surf
        return sat_surf


    def _calc_corner_ad(self, pc, phys_water, case, occupy, **kwargs):
        wtr = phys_water
        element = ['pore', 'throat']
        prop = '.corner_area_'+case
        for item in element:
            trap_pc = '.'.join([item, case+'_trapped_pc'])
            wtr.models.add(propname=item+prop,
                           model=pm.corner_adsorption.area,
                           pc=pc, trapping=self._trapping,
                           trap_pc=trap_pc,
                           film_thickness=item+'.film_thickness_'+case,
                           film_area=item+'.film_area_'+case,
                           corner_area=item+prop,
                           pore_occupancy=item+'.'+occupy)
#        wtr.models.regenerate()
#        temp = _sp.argsort(wtr._net['throat.shapefactor'])
#        Ac = wtr['throat'+prop][temp]
#        print('pc \t {}, Acmin \t {:.3g}, Acmax \t {:.3g}'.format(pc, Ac[0],
#              Ac[-1]))
#        if pc > -4000:
#            print(str(len(_sp.where(wtr['pore'+prop]))))
        volume = 0
        prop2 = '.corner_volume_'+case
        for item in element:
            wtr.models.add(propname=item+prop2,
                           model=pm.corner_adsorption.volume,
                           pore_occupancy=item+'.'+occupy,
                           corner_area=item+prop)
            volume += sum(wtr[item+prop2])

        sat_corner = volume/(self._vol_total)
        self._sat_corner[case].append(sat_corner)
        self._sat[case][-1] += sat_corner   # update total saturations
#        Results[case]['sat_corner'].append([pc, sat_corner])
#        Results[case]['saturation'][-1][1] += sat_corner
        return sat_corner


    def _adjust_occupancy(self, water, case, occupy):
        r'''
        Adjust the occupancy (condensation based) by the adsorption-based occ
        '''
        element = ['pore', 'throat']
        prop1 = '.film_area_' + case
        prop2 = '.corner_area_' + case
        prop3 = '.' + occupy
        for item in element:
            Afilm = water[item+prop1]
            Acorner = water[item+prop2]
            Asurface = Afilm + Acorner

            Apore = self._net[item+'.area']
#            occ_surf = Asurface >= Apore
            occ_surf = _sp.isclose(Asurface-Apore, 0, atol=1e-20)
            occ_surf += (Asurface >= Apore)
#            if _sp.any(occ_surf):
            water[item+prop3] += 1*occ_surf

    def _vapour_conductance(self, pc, phys_vapour, case, **kwargs):
        vpr = phys_vapour
        prop = 'throat.diffusive_conductance_'+case
        vpr.models.add(propname=prop,
                       model=pm.diffusive_conductance.bulk_diffusion,
                       pc=pc, knudsen=self._knudsen,
                       film_thickness='throat.film_thickness_'+case,
                       film_area='throat.film_area_'+case,
                       corner_area='throat.corner_area_'+case)
        vpr.models.add(propname=prop+'_pore',
                       model=pm.diffusive_conductance.bulk_diffusion,
                       pc=pc, knudsen=self._knudsen,
                       film_thickness='pore.film_thickness_'+case,
                       film_area='pore.film_area_'+case,
                       corner_area='pore.corner_area_'+case)
#        vpr.models.regenerate()
#        temp = _sp.argsort(vpr._net['throat.shapefactor'])
#        gv = vpr[prop][temp]
#        print('pc \t {}, gvmin \t {:.3g}, gvmax \t {:.3g}'.format(pc, gv[0],
#              gv[-1]))


    def _surface_conductance(self, pc, phys_water, case, occupy, **kwargs):
        wtr = phys_water
        prop = 'throat.surface_conductance_'+case
        wtr.models.add(propname=prop,
                       model=pm.hydraulic_conductance.surface_cond,
                       base='shapefactor', #radius, shapefactor
                       film_area='throat.film_area_'+case,
                       corner_area='throat.corner_area_'+case,
                       pore_occupancy='throat.'+occupy)
        wtr.models.add(propname=prop+'_pore',
                       model=pm.hydraulic_conductance.surface_cond,
                       base='shapefactor',
                       film_area='pore.film_area_'+case,
                       corner_area='pore.corner_area_'+case,
                       pore_occupancy='pore.'+occupy)
#        wtr.models.regenerate()
#        temp = _sp.argsort(wtr._net['throat.shapefactor'])
#        gsurf = wtr[prop][temp]
#        print('pc \t {}, gvmin \t {:.3g}, gvmax \t {:.3g}'.format(pc, gsurf[0],
#              gsurf[-1]))


    def _conduit_conductance(self, pc, phys_moisture, case, occupy,
                             **kwargs):
        if self._surf_flow or self._corner_flow:
            sflow = True
        else:
            sflow = False
        if sflow:
            model = pm.multiphase.mixed_surf_conduc
        else:
            model = pm.multiphase.mixed_conductance

        diffprop = 'throat.diffusive_conductance_'+case
        surfprop = 'throat.surface_conductance_'+case
        phys_moisture.models.add(propname='throat.conduit_conductance_'+case,
                                 model=model,
                                 diffusion=self._diffusion,
                                 throat_occupancy='throat.'+occupy,
                                 pore_occupancy='pore.'+occupy,
                                 tdiffusive_conductance=diffprop,
                                 tsurf_diff_cond=surfprop,
                                 surface_flow=sflow)
#        phys_moisture.models.regenerate()
#        temp = _sp.argsort(phys_moisture._net['throat.shapefactor'])
#        g = phys_moisture['throat.conduit_conductance_'+case][temp]
#        print('pc \t {}, gmin \t {:.3g}, gmax \t {:.3g}'.format(pc, g[0],
#              g[-1]))


    def _conduit_single_conductance(self, pc, phys_water, phys_vapour,
                                   case, occupy, **kwargs):
        hydprop = 'throat.hydraulic_conductance'
        diffprop = 'throat.diffusive_conductance_'+case
        surfprop = 'throat.surface_conductance_'+case
#        if case == 'drying_wetting':
#            case
        phys_water.models.add(propname='throat.conduit_conductance_'+case,
                              model=pm.multiphase.single_conductance_pore,
                              factor=1e-40,
                              throat_occupancy='throat.'+occupy,
                              pore_occupancy='pore.'+occupy,
                              pconductance=hydprop+'_pore',
                              tconductance=hydprop,
                              psurfcond=surfprop+'_pore',
                              tsurfcond=surfprop)
        phys_vapour.models.add(propname='throat.conduit_conductance_'+case,
                               model=pm.multiphase.single_conductance_pore,
                               factor=1e-40,
                               throat_occupancy='throat.'+occupy,
                               pore_occupancy='pore.'+occupy,
                               pconductance=diffprop+'_pore',
                               tconductance=diffprop)


    def _modify_solver(self, p_occ, t_occ, **kwargs):
        r"""
        Arguments:
        wet_ratios: array_like of classification boundaries, of size n.
        amg_solvers, tolerances, strengths, CFs, aggs, smooths.

        Return:
        New modified kwargs
        """
        wet_ratio = (p_occ.sum() + t_occ.sum())/(self._net.Np+self._net.Nt)
        klas = _sp.searchsorted(self._thresholds, wet_ratio)

        kwargs['amg'] = self._amg_solvers[klas]
        kwargs['tol'] = self._tolerances[klas]
        kwargs['strength'] = self._strengths[klas]
        kwargs['CF'] = self._CFs[klas]
        kwargs['agg'] = self._aggs[klas]
        kwargs['smooth'] = self._smooths[klas]
#        kwargs['accel'] = 'gmres'
        return kwargs


    def _max_cluster(self, span, dead, isolated, **kwargs):
        try:
            spanmax = span[0].size
        except:
            spanmax = 0
        try:
            deadmax = dead[0].size
        except:
            deadmax = 0
        try:
            isomax = isolated[0].size
        except:
            isomax = 0

        nmax = _sp.amax([spanmax, deadmax, isomax])
        return nmax


    def _calc_flow(self, phase, pc, case, bound, pc1, pc2, kphase,
                   save_matrix=False, modify=True, row_scaling=False,
                   perm_error=False, singleflow=False,
                   name=None, **kwargs):
        FlowClass = MoistureFlow
        if name is None:
            name = 'alg_flow'
        if phase.name == 'water':
            FlowClass = WaterFlow
        elif phase.name == 'vapour':
            FlowClass = WaterFlow
        name += '_' + case
        pn = self._net

        alg_flow = FlowClass(name=name, network=pn, phase=phase)

        if self._is_span:
            BC1_pores = pn.pores(labels=bound[0])
            BC2_pores = pn.pores(labels=bound[1])

            # BC1
            alg_flow.set_boundary_conditions(bctype='Dirichlet', bcvalue=pc1,
                                             pores=BC1_pores)
            # BC2
            alg_flow.set_boundary_conditions(bctype='Dirichlet', bcvalue=pc2,
                                             pores=BC2_pores)

            # run algorithms with proper conduit conductance
            alg_flow.setup(conductance='conduit_conductance_'+case,
                          quantity='pressure_'+case, **kwargs)  # create matrix A

            # ==== save the original matrix ===
    #        import os
    #        name_mat = 'reg'
    #        folder = os.getcwd()
    #        A_name = folder + '/A_' + name_mat
    #        b_name = folder + '/b_' + name_mat
    #        bpnm.Utilities.IO.save_sparse_csr(A_name, alg_flow.A)
    #        _sp.save(b_name, alg_flow.b)
            #=====================================
            if modify:
                if self._modified and not self._single_flow:
                    indices = self._A_indices
                    indptr = self._A_indptr
                    shape = self._A_shape
                else:
                    indices, indptr, shape = None, None, None

                alg_flow._modify_system(indices=indices, indptr=indptr,
                                        shape=shape, row_scaling=False, **kwargs)
                if not self._modified and not self._single_flow:
                    self._A_indices = alg_flow.A.indices.copy()
                    self._A_indptr = alg_flow.A.indptr.copy()
                    self._A_shape = copy.deepcopy(alg_flow.A._shape)
                    self._modified = True
                if row_scaling:
                    alg_flow._row_scaling()

            res = []

            btime = time.time()         # wall-clock time [s]
            atime = time.perf_counter() # cpu time [s]

            if not perm_error:

                alg_flow.solve(x0=pc, res=res, **kwargs)

                # calc effective permeabilities [s]
                conductance = phase['throat.conduit_conductance_' + case]
                # print(conductance)
                k_eff = alg_flow.calc_eff_permeability(
                    conductance=phase['throat.conduit_conductance_'+case])
            else:

                perm_rel_err = []
                k_eff = self._permeability_rel_error(alg_flow, pc, res, case,
                                                     kphase, perm_rel_err,
                                                     **kwargs)

            if pc == -880:
                print('pc: {}'.format(str(pc)))
            time_solver = time.perf_counter() - atime
            time_solver2 = time.time() - btime

            alg_flow.return_rate(case=case, debug_values=DEBUG, pc=pc, **kwargs)

        else:
            k_eff = [0, 0, 0, 0]

        ctrl.purge_object(alg_flow)
        # append permeability & flow values to the lists
        self._eff_permeability[case][kphase].append(k_eff[0])
        self._eff_permeability2[case][kphase].append(k_eff[1])
        self._eff_permeability_vap[case][kphase].append(k_eff[2])
        self._eff_permeability_vap2[case][kphase].append(k_eff[3])

        # record the result: for parallel
        Results[case]['permeability'][kphase].append([pc, k_eff[:2]])
        Results[case]['permeability_vap'][kphase].append([pc, k_eff[2:]])
#        print('eff permeability = {} \t at pc = {}'.format(k_eff, pc))


        if save_matrix and self._is_span:
            self._save_matrices(alg_flow.A, alg_flow.b, pc, **kwargs)

        if self._debug and self._is_span:
            self._res[case].append(res)
            twater = self._water._physics[0]['throat.hydraulic_conductance']
            pwater = self._water._physics[0]['throat.hydraulic_conductance_pore']
            tvapour = self._phys_vapour['throat.diffusive_conductance_'+case]
            pvapour = self._phys_vapour['throat.diffusive_conductance_'+case+'_pore']
            p1 = self._net['throat.conns'][:,0] # Nt long
            p2 = self._net['throat.conns'][:,1]
            p1_occ = kwargs['p_occ'][p1].astype('bool')    # Nt long
            p2_occ = kwargs['p_occ'][p2].astype('bool')

            c_water = _sp.r_[twater[kwargs['t_occ'].astype('bool')],
                             pwater[:,0][p1_occ], pwater[:,1][p2_occ]]
            c_vapour = _sp.r_[tvapour[~kwargs['t_occ'].astype('bool')],
                              pvapour[:,0][~p1_occ], pvapour[:,1][~p2_occ]]
            try:
                c_watermin = c_water.min()
                c_watermax = c_water.max()
            except:
                c_watermin = 0
                c_watermax = 0
            try:
                c_vapourmin = c_vapour.min()
                c_vapourmax = c_vapour.max()
            except:
                c_vapourmin, c_vapourmax = 0, 0

            self._cwater_min[case].append(c_watermin)
            self._cwater_max[case].append(c_watermax)
            self._cvapour_min[case].append(c_vapourmin)
            self._cvapour_max[case].append(c_vapourmax)
            self._BCplane[case].append(alg_flow._BCplane[0])
            self._flow2[case].append(alg_flow._flow2[0])
            self._flow[case].append(alg_flow._flow[0])
#            self._time_solver[case].append(time_solver)
#            self._solver_list[case].append(kwargs['amg'])
#            self._N_iters[case].append(len(res))
#            self._tols[case].append(kwargs['tol'])
            try:
                norm_b = _sp.linalg.norm(alg_flow.b)
                self._norm_b[case].append(norm_b)
                self._rel_residues[case].append(res[-1]/norm_b)
            except:
                self._rel_residues[case].append([])     # if amg=None
        if self._keep_log and self._is_span:
            self._time_solver[case].append(time_solver)
            self._time_solver_wall[case].append(time_solver2)
            self._solver_list[case].append(kwargs['amg'])
            self._N_iters[case].append(len(res))
            self._tols[case].append(kwargs['tol'])
            try:
                self._solver_list[case][-1] += '_' + kwargs['accel']
            except:
                pass
            if perm_error:  # revise tolerance records
                self._solver_list[case][-1] += '_k_err'
                try:
                    self._tols[case][-1] = kwargs['perm_tol']
                except:
                    self._tols[case][-1] = 1e-3
        try:
            self._par_pc
            return alg_flow, k_eff
        except:
            return alg_flow

    def _calc_single_flow(self, phase, pc, case, bound, pc1, pc2, direction,
                          p_occ, t_occ,
                          save_matrix=False, modify=True, row_scaling=False,
                          perm_error=False, phase_pocc=None, **kwargs):
        pn = self._net
        # ALGORITHMS (assumption: no surface flow)
        # 1. check the water and air spanning clusters
        # 2. create both spanning water and air networks (trimmed isolated and non-phase pores)
        # 3. recalculate the phase's conduit conductances
        # 4. run the permeabilty calculation
        # IF with surface flow, the water cluster is always connected.
        # In hygroscopic region, the water and vapour flows coexist.
        # No water/air cluster identification required, just set the conduit conductance to zero
        # and if the pressure solver fails set the permeability to zero.
        tmask = ~_sp.bool_(t_occ)
        try:
            (aSpanBool, aSpanCl, aIsol) = pn.span_existence(mask=tmask)
        except:
            (aSpanBool, aSpanCl) = pn.span_existence(mask=tmask)

#        if p_occ_wp.max():
        try:
            (SpanBool, SpanCl, Isol) = pn.span_existence(mask=p_occ_wp)
        except:
            (SpanBool, SpanCl) = pn.span_existence(mask=p_occ)


        one_flow = WaterFlow(name=phase.name+'_flow_'+case, network=pn,
                             phase=phase)

        if phase_pocc is not None:
            (SpanBool, SpanCl) = pn.span_existence(mask=phase_pocc,
                                                   return_isolated=False)
            if not SpanBool:
                print('The spanning {} cluster does not exist'.format(phase.name))
                self._eff_permeability[case][direction].append(0)
                self._eff_permeability2[case][direction].append(0)
                return one_flow

        BC1_pores = pn.pores(labels=bound[0])
        BC2_pores = pn.pores(labels=bound[1])

        # BC1
        one_flow.set_boundary_conditions(bctype='Dirichlet', bcvalue=pc1,
                                         pores=BC1_pores)
        # BC2
        one_flow.set_boundary_conditions(bctype='Dirichlet', bcvalue=pc2,
                                         pores=BC2_pores)

        # run algorithms with proper conduit conductance
        one_flow.setup(conductance='conduit_conductance_'+case,
                      quantity='pressure_'+case, **kwargs)
        if modify:
            if self._modified:
                indices = self._A_indices
                indptr = self._A_indptr
                shape = self._A_shape
            else:
                indices, indptr, shape = None, None, None

            one_flow._modify_system(indices=indices, indptr=indptr,
                                    shape=shape, row_scaling=False, **kwargs)
            if not self._modified:
                self._A_indices = one_flow.A.indices.copy()
                self._A_indptr = one_flow.A.indptr.copy()
                self._A_shape = copy.deepcopy(one_flow.A._shape)
                self._modified = True
            if row_scaling:
                one_flow._row_scaling()

        res = []
        atime = time.clock()
        if not perm_error:
            one_flow.solve(x0=pc, res=res, **kwargs)
            # calc effective permeabilities [s]
            k_eff = one_flow.calc_eff_permeability(
                conductance=self._phys_moisture['throat.conduit_conductance_'+case])
        else:
            perm_rel_err = []
            k_eff = self._permeability_rel_error(one_flow, pc, res, case,
                                                 direction, perm_rel_err,
                                                 **kwargs)
        time_solver = time.clock() - atime

        # append permeability & flow values to the lists
        self._eff_permeability[case][direction].append(k_eff[0])
        self._eff_permeability2[case][direction].append(k_eff[1])

        print('eff permeability = {} \t at pc = {}'.format(k_eff, pc))

        one_flow.return_rate(case='_'+case)
        ctrl.purge_object(one_flow)
        try:
            self._par_pc
            return one_flow, k_eff
        except:
            return one_flow


    def _permeability_rel_error(self, alg_flow, pc, res, case, direction,
                                perm_rel_err, maxiter=100, perm_tol=1e-3,
                                tol=1e-14, **kwargs):
        tol = 1e-14
        k_old = 0
        alg_flow.X = _sp.ones_like(alg_flow.b)*pc
        perm_rel_err.append(1)
        while len(perm_rel_err) <= maxiter and perm_rel_err[-1] > perm_tol:
            resi = []
            alg_flow.solve(x0=alg_flow.X, res=resi, maxiter=1, tol=tol,
                           **kwargs)

            # calc effective permeabilities [s]
            k_new = alg_flow.calc_eff_permeability(
                conductance=self._phys_moisture['throat.conduit_conductance_'+case])

            perm_rel_err.append(_sp.absolute(k_new-k_old)/_sp.absolute(k_new))
            k_old = k_new
            res.extend(resi[1:])
        return k_old


    def _save_matrices(self, A, b, pc, pcs=[], ws=[], folder=None,
                       name_mat='', **kwargs):
        if folder is None:
            import os
            folder = os.getcwd()
#            folder = '/home/islah/Documents/python3/multiscale/regular_multiscale_net/'
        A_name = folder + '/A_' + name_mat
        b_name = folder + '/b_' + name_mat
        for i in pcs:
            if _sp.absolute(pc-i) < 2:
                i = _sp.around(_sp.log10(-i), decimals=1)
                bpnm.Utilities.IO.save_sparse_csr(A_name+'_lpc'+str(i), A)
                _sp.save(b_name+'_lpc'+str(i), b)

#        for i in ws:
#            if _sp.absolute(w-i) < 0.1:
#                bpnm.Utilities.IO.save_sparse_csr(A_name+'_w'+str(i), A)
#                _sp.save(b_name+'_w'+str(i), b)


    def _grid_refinement(self, case, pc_list, dsat_threshold=0.2,
                         dperm_threshold=3, **kwargs):
        r"""
        Change and replace pc_list argument
        """
        arg = self._sorting(case)

        n_rest = len(pc_list)
        self._pc_grids = self._pc_grids[n_rest:][arg]  # reverse the order
        if n_rest > 0:
            pc_list = []

        # refine based on the moisture content:
        # this is better to evenly divide the grid in hysteresis k(w)
        dsat = _sp.absolute(_sp.diff(self._sat[case]))
        tag = _sp.where(dsat>dsat_threshold)[0]
        tag = _sp.unique(tag)

#        # refine based on the permeability
#        dperm = _sp.absolute(_sp.diff(_sp.log10(self._eff_permeability[case][0])))
#        tag = _sp.where(dperm>dperm_threshold)[0]
#        tag = _sp.unique(tag)

        pcs = _sp.array(self._pcs[case])
        tagged = _sp.array([pcs[tag], pcs[tag+1]])
        fine = -_sp.power(10, _sp.mean(_sp.log10(-tagged), axis=0))
        fine = _sp.round_(fine)
        if len(tag)>0:
            pc_list.extend(fine)
            self._pc_grids = _sp.concatenate((self._pc_grids,fine[::-1]))

        return pc_list

    def _sort_result(self, d):
        return d[_sp.argsort(d[:, 0])]


    def _sorting_par(self, result, case):
        result = self._sort_result(_sp.array(result))

        self._pcs[case] = list(result[:, 0])
        self._sat[case] = list(result[:, 1])
        if self._surface_ad:
            self._sat_surf[case] = list(result[:, 2])
        if self._corner_ad:
            self._sat_corner[case] = list(result[:, 3])
        keff = self._eff_permeability[case]
        keff2 = self._eff_permeability2[case]
        kveff = self._eff_permeability_vap[case]
        kveff2 = self._eff_permeability_vap2[case]
        for key in keff.keys():
            if key == 'moisture':
                keff[key] = list(result[:, 4])
                keff2[key] = list(result[:, 5])
                kveff[key] = list(result[:, 6])
                kveff2[key] = list(result[:, 7])
        self._alg_flow = {case: self._alg_flow}


    def _sorting(self, case):
        r"""
        Sorting and modifying self._pcs, _sat, eff_permeability[case][keys], and
        sat_surf when applicable.

        Note: this function replaces the old lists with the new ones!
        """

        #sort pcs, saturation and permeability arrays
        arg = _sp.argsort(self._pcs[case])
        self._pcs[case] = list(_sp.array(self._pcs[case])[arg])
        self._sat[case] = list(_sp.array(self._sat[case])[arg])

        keff = self._eff_permeability[case]
        keff2 = self._eff_permeability2[case]
        kveff = self._eff_permeability_vap[case]
        kveff2 = self._eff_permeability_vap2[case]
        for key in keff.keys():
            if keff[key]:
                keff[key] = list(_sp.array(keff[key])[arg])
                keff2[key] = list(_sp.array(keff2[key])[arg])
                kveff[key] = list(_sp.array(kveff[key])[arg])
                kveff2[key] = list(_sp.array(kveff2[key])[arg])
        if self._surface_ad:
            self._sat_surf[case] = list(_sp.array(self._sat_surf[case])[arg])
        if self._corner_ad:
            self._sat_corner[case] = list(_sp.array(self._sat_corner[case])[arg])
        return arg


    def _store_result(self, case, **kwargs):
        r"""
        Store alg_flow results to internal variables
        """
        if not self._surface_ad:
            sat_surf = []
        else:
            sat_surf = self._sat_surf[case]

        if not self._moist_vol:
            sat_moist = []
        else:
            sat_moist = self._sat_moist[case]

        keys = ['pc', 'sat', 'sat_surf', 'sat_moist', 'w_sat']
        vals = [self._pcs[case], self._sat[case], sat_surf, sat_moist,
                self._w_sat]
        keff = self._eff_permeability[case]
        keff2 = self._eff_permeability2[case]
        kveff = self._eff_permeability_vap[case]
        kveff2 = self._eff_permeability_vap2[case]
        for key in keff.keys():
            keys.append('k_'+key)
            vals.append(keff[key])
            keys.append('k2_'+key)
            vals.append(keff2[key])
            keys.append('deltav_'+key)
            vals.append(kveff[key])
            keys.append('deltav2_'+key)
            vals.append(kveff2[key])
        results = {k:v for (k,v) in zip(keys, vals)}

        try:
            self._par_pc
        except:
            self._alg_flow[case].result = results

            self._alg_flow[case].store_result(Pc=self._pcs[case],
                                              sat=self._sat[case],
                                              sat_surf=sat_surf,
                                              sat_moist=sat_moist,
                                              w_sat=self._w_sat,
                                              k=self._eff_permeability[case],
                                              k2=self._eff_permeability2[case],
                                              dv=self._eff_permeability_vap[case],
                                              dv2=self._eff_permeability_vap2[case])


    def plot(self, **kwargs):
        r"""
        Plot all cases
        """
        cases = [case.split('_') for case in self._cases]
        cases = list(set([case for sublist in cases for case in sublist]))
        if 'drying' in cases:
            cases.remove('drying')
            for case in cases:
                self._plot_hysteresis(case=case)
        else:
            for case in cases:
                self._plot_1case(case=case)


    def _plot_hysteresis(self, case, **kwargs):
        # the 'case' here is either 'wetting' or 'imbibition'
        alg_flow_wp = self._alg_flow[case]
        alg_flow_dp = self._alg_flow['drying_'+case]
        try:
            bpnm.Postprocessing.Plots.hysteresis(alg_flow_wp, alg_flow_dp,
                                                 legend=[case, 'drying'])
        except:
            data = self.create_data()
            bpnm.Postprocessing.Plots.hysteresis_from_dict(data)


    def _plot_1case(self, case, **kwargs):
        bpnm.Postprocessing.Plots.plot_wp(self._alg_flow[case], name=case)


    def create_data(self, cases=None):
        r'''
        Create a dictionary of the calculated hygric properties
        with Npc long arrays.
        '''
        from bwfpnm.Phases.models import vapour_pressure as fpv
        if cases is None:
            cases = self._pcs.keys()
        for item in self._net._phases:
            if item.name == 'vapour':
                phase = item
                break
        data = {}
        for case in cases:
            pv, RH = fpv.pore(phase, self._pcs[case], with_RH=True)
            data[case] = {'pc': self._pcs[case],
                          'lpc': _sp.log10(-_sp.array(self._pcs[case])),
                          'RH': RH,
                          'pv': pv,
                          'sat total': self._sat[case],
                          'moisture content': self._moisture_content[case]}

            kphases = self._eff_permeability[case]
            kphases2 = self._eff_permeability2[case]
            kvphases = self._eff_permeability_vap[case]
            kvphases2 = self._eff_permeability_vap2[case]
            for key in kphases.keys():
                data[case].update({'k_'+str(key): kphases[key],
                                   'k_'+str(key)+'2': kphases2[key],
                                   'deltav_'+str(key): kvphases[key],
                                   'deltav_'+str(key)+'2': kvphases2[key]})

            try:
                data[case].update({'sat film': self._sat_surf[case]})
            except:
                pass
            try:
                data[case].update({'sat corner': self._sat_corner[case]})
            except:
                pass

            if self._debug:
                data[case].update({'N wet pores': self._pocc_number[case],
                            'N wet throats': self._tocc_number[case],
                            'N spanning cluster': self._span_number[case],
                            'N surface cluster': self._dead_number[case],
                            'N isolated cluster': self._isolated_number[case],
                            'biggest-cluster size': self._max_cluster_size[case],
                            'max_water_conductance': self._cwater_max[case],
                            'min_water_conductance': self._cwater_min[case],
                            'max_vapour_conductance': self._cvapour_max[case],
                            'min_vapour_conductance': self._cvapour_min[case],
                            'Rel residues': self._rel_residues[case],
                            'Norm b': self._norm_b[case],
                            'Flow in': self._flow[case],
                            'Flow out': self._flow2[case]})
    #                        'Flow': self._flow[case]})
            if self._keep_log:
                data[case].update({'solving_time': self._time_solver[case],
                                   'solving_time_wall': self._time_solver_wall[case],
                                   'solvers': self._solver_list[case],
                                   'N iters': self._N_iters[case],
                                   'Tolerances': self._tols[case]})
#        cases = _sp.array(cases, ndmin=1)
#        if cases.size == 1:
#            data = data[case]
        return data

#    def return_residual(self, pores=None, throats=None, case='', **kwargs):
#        r"""
#        Send residual of pressure result to phase.
#        """
#        if pores is None:
#            pores = self.Ps
#        if throats is None:
#            throats = self.Ts
#
#        phase_quantity = self._quantity.replace(self._phase.name + '_', '')
#        if phase_quantity not in self._phase.props():
#            self._phase[phase_quantity] = _sp.nan
#        self._phase[phase_quantity][pores] = self[self._quantity][pores]
#        conn_arr = self._net.find_connected_pores(self.Ts)
#        dx = _sp.squeeze(_sp.diff(self[self._quantity][conn_arr], n=1, axis=1))
#        g = self['throat.conductance']
#        rate = _sp.absolute(g * dx)
#        if 'throat.rate' not in self._phase.props():
#            self._phase['throat.rate'+case] = _sp.nan
#        self._phase['throat.rate'+case][throats] = rate[throats]
#        self._phase['throat.delta_pressure'+case] = dx
#        logger.debug('Results of ' + self.name +
#                     ' algorithm have been added to ' + self._phase.name)

#    def permeability_curve_parallel(self, cases=['wetting'], w_sat=1,
#                           trapping=False, num_seq=10,
#                           knudsen=True, plot=True, printstatus=False,
#                           surface_ad=False, moist_volume=False, dPc =1,
#                           diffusion=True, refine=False, debug=False,
#                           moist_vol=False, **kwargs):
#        r"""
#        Parallel version of permeability_curve().
#        """
#        self._setup(knudsen, surface_ad, diffusion, refine, plot, printstatus,
#                    dPc, debug, moist_vol, w_sat, **kwargs)
#        self._calc_pc_grids(num_seq)
#        self._define_cases(cases, trapping)
#        if surface_ad:
#            self._sat_surf = {}
#        if moist_vol:
#            self._sat_moist = {}
#
#        for case in self._cases:
#            if case in ['wetting', 'imbibition']:
#                occupancy = self._occupancy[case]
#                occupy = self._occupy[case]
#                alg = self._alg_wp
#            elif case == 'drying_wetting':
#                occupancy = self._occupancy['drying']['wetting']
#                occupy = occupancy
#                alg = self._alg_dp
#            elif case == 'drying_imbibition':
#                occupancy = self._occupancy['drying']['imbibition']
#                occupy = occupancy
#                alg = self._alg_dp
#
#            self._sat[case] = []
#            self._pcs[case] = []
#            self._eff_permeability[case] = {0: []}
#            if surface_ad:
#                self._sat_surf[case] = []
#            if moist_vol:
#                self._sat_moist[case] = []
#            self._i = 0
#            pc_list = self._pc_grids.copy()
##            case_split = case.split('_')
#            case_split = [case]
#            self._alg = alg
##            self._pc = pc
#            self._case = case
#            self._case_split = case_split
#            self._occupancy_i = occupancy
#            self._occupy_i = occupy
#
##            while len(pc_list) > 0:
##                pc = pc_list.pop()
#
#            argums = [(pc, )\
#                     for pc in pc_list[::-1]]
#            pool = Pool(cpu_count()-1)
#            pool.map(self.iterations, argums)
#            pool.close()
#            pool.join()
#
#            self._alg_flow[case].calc_abs_permeability()
#            self._alg_flow[case].calc_mD_permeability()
#
#            self._store_result(case)
#            self._moisture_content[case] = _sp.array(self._sat[case])*w_sat
#
#        if plot:
#            self.plot()
#
#
#    def iterations(self, arg):
#        pc = arg
#        self._alg.return_results(Pc=pc, cases=self._case_split,
#                           occupancies=[self._occupancy_i],
#                           trapping=self._trapping)
#        self._alg_flow[case] = self._calc_each_pc(pc, self._case,
#                                                  self._occupy_i)
##        if self._sat[case][-1] >= 1:
##            self._sat[case].append(self._sat[case][-1])
##            self._pcs[case].append(pc_list.pop())
##            k = self._eff_permeability[case][0]
##            k.append(k[-1])
##            break
#        self._i += 1
#        return 1



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

    #%% Algorithm
    case = 'wetting'
    amg = None
    alg_wp, alg_dp, w_sat, porosity = bwfr.moist_content(geo, alg_wp,
                                                     water['pore.density'][0],
                                                     alg_dp=alg_dp,
                                                     cases=[case],
                                                     trapping=False)

    # %% Permeability
    perm = Permeability(pn, alg_wp, phys_vapour, phys_moisture,
                        alg_dp=alg_dp)
    perm.permeability_curve(cases=['wetting'], w_sat=w_sat,
                            trapping=False, num_seq=50,
                            knudsen=True, plot=True, printstatus=False,
                            surface_ad=False, moist_volume=False, dPc =1,
                            diffusion=True, refine=True, keep_log=False,
                            amg=amg, tol=1e-14,
                            strength=('symmetric', {'theta': 0.03}),
                            CF='RS', agg='standard', smooth='energy')

    bpnm.Postprocessing.Plots.plot_2scales(-_sp.array(perm._pcs[case]),
                                           perm._moisture_content[case],
                                           perm._eff_permeability[case][0])


