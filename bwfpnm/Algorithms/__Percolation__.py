# -*- coding: utf-8 -*-
"""
===============================================================================
ImbibitionPercolation: modified OpenPNM's InvasionPercolation
===============================================================================

"""
import heapq as hq
import scipy as sp
import numpy as np
from OpenPNM.Algorithms import GenericAlgorithm
from OpenPNM.Base import logging
from collections import Counter as counter
from bwfpnm.Utilities import isclose
from scipy.sparse import csr_matrix, lil_matrix
from .find_trapped_pores import _find_trapped_pores
import time

logger = logging.getLogger(__name__)


class Percolation(GenericAlgorithm):
    r"""
    A classic/basic invasion percolation algorithm optimized for speed.

    Parameters
    ----------
    network : OpenPNM Network object
        The Network upon which the invasion should occur.

    Notes
    ----
    n/a

    """

    def __init__(self, phase_wet=None, phase_dry=None, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self._phase_wet = phase_wet
        self._phase_dry = phase_dry
        self._Mseq = (self.Np+self.Nt)*1000
        logger.debug("Create Percolation Algorithm Object")

        phase = self._phase_wet
        self['throat.entry_pressure'] = phase['throat.capillary_pressure']
        self['pore.entry_pressure'] = phase['pore.capillary_pressure']
        self['throat.sorted'] = sp.argsort(self['throat.entry_pressure'], axis=0)
        self['pore.sorted'] = sp.argsort(self['pore.entry_pressure'], axis=0)
        self._eps = eps

    def setup_imbibition(self, **kwargs):
        r"""
        Set up the required parameters for the algorithm:
        pore/throat.imbibition_order

        Initialize: pore/throat.inv_sequence = -1
        """
        order = 'imbibition_order'
        # Throat order based on the entry pressure
        # note: self['throat.order'][min, max entry pressure] = 0, Nt
        self['throat.'+order] = sp.zeros_like(self['throat.sorted'])
        self['throat.'+order][self['throat.sorted']] = sp.arange(0, self.Nt)
        self['pore.'+order] = sp.zeros_like(self['pore.sorted'])
        self['pore.'+order][self['pore.sorted']] = sp.arange(0, self.Np)
        self._tcount = 0

    def setup_wetting(self, inv_points=None, **kwargs):
        r"""
        Set up the required parameters for the algorithm:
        self._wetting_pc: Npc long
        self._p_order: Npc long
        self._t_order: Npc long

        Initialize: pore/throat.inv_sequence = -1

        Parameters
        ----------
        phase : OpenPNM Phase object
            The phase to be injected into the Network.  The Phase must have the
            capillary entry pressure values for the system.

        """

        # Setup arrays and info
        t_pc = self['throat.entry_pressure']
        p_pc = self['pore.entry_pressure']
        # Sorted unique entry pressures
        if inv_points is None:
            # inv_points=all pc, use equality sign
            self._wetting_pc = sp.unique(sp.r_[t_pc, p_pc]) #already sorted
            self._p_order = sp.array([[] if len(sp.where(isclose(p_pc, pc))[0])==0\
                else sp.where(isclose(p_pc, pc))[0] for pc in self._wetting_pc])
            self._t_order = sp.array([[] if len(sp.where(isclose(t_pc, pc))[0])==0\
                else sp.where(isclose(t_pc, pc))[0] for pc in self._wetting_pc])
        else:
            # use '<=' sign
            inv_points = sp.unique(inv_points)  # unique sorted array of pc
            _p_order = [[]]*len(inv_points)
            _t_order = [[]]*len(inv_points)
            for i, pc in enumerate(inv_points):
                pmask = p_pc <= pc
                tmask = t_pc <= pc
                if i > 0:
                    _p_order[i] = self.toindices(pmask*~pmask_all)
                    _t_order[i] = self.toindices(tmask*~tmask_all)
                    pmask_all += pmask
                    tmask_all += tmask
                else:
                    _p_order[i] = self.toindices(pmask)
                    _t_order[i] = self.toindices(tmask)
                    pmask_all = pmask
                    tmask_all = tmask
            self._p_order = sp.array(_p_order)
            self._t_order = sp.array(_t_order)
            self._wetting_pc = inv_points

        # not implemented like below, since size != Np or Nt
#        self['pore.wetting_order'] = _p_order
#        self['throat.wetting_order'] = _t_order
        self._tcount = 0

    def setup_drying(self, p_inv_pc=None, t_inv_pc=None, case='imbibition',
                     pore_inv_pc='pore.imbibition_inv_pc_trapping',
                     throat_inv_pc='throat.imbibition_inv_pc_trapping',
                     **kwargs):
        r"""
        Set up the required parameters for the algorithm:
        pore/throat.drying_+case+_order

        Parameters
        ----------
        p_inv_pc : array
            The array of pore's pc invasion, as produced by wetting/imbibition.
            This is used to determine the initial condition of moisture
            distribution.

        """
        # Initial conditions: moisture distribution
        if p_inv_pc is None:
            try:
                self._pwet = self[pore_inv_pc]<0
                self._twet = self[throat_inv_pc]<0
            except:
                self._pwet = True
                self._twet = True
        else:
            self._pwet = p_inv_pc<0
            self._twet = t_inv_pc<0
        self._p_inv_pc = p_inv_pc
        self._t_inv_pc = t_inv_pc

        order = 'drying_'+case+'_order'
        # Throat order based on the entry pressure
        # note: self['throat.order'][min, max entry pressure] = 0, Nt
        self['throat.'+order] = sp.zeros_like(self['throat.sorted'])
        sort = self['throat.sorted'][::-1]
        self['throat.drying_sorted'] = sort
        self['throat.'+order][sort] = sp.arange(0, self.Nt)
        self['pore.'+order] = sp.zeros_like(self['pore.sorted'])
        sort = self['pore.sorted'][::-1]
        self['pore.drying_sorted'] = sort
        self['pore.'+order][sort] = sp.arange(0, self.Np)
        self._tcount = 0

    def set_inlets_drying(self, pores=None, case='imbibition', **kwargs):
        r"""

        Parameters
        ----------
        pores : array_like
            The list of inlet pores from which the Phase can enter the Network
        """
        if 'inlets' in kwargs.keys():
            pores = kwargs['inlets']
        elif pores is None:
            pores = self._net['pore.outlet'] # + self._net['pore.inlet']

        self._inlet_drying = pores
        self._net['pore.inlet_drying_'+case] = pores
        # Perform initial analysis on input pores
        self.Pqueue = []
        inpores = self['pore.drying_'+case+'_order'][pores*self._pwet]
        [hq.heappush(self.Pqueue, P) for P in inpores]

    def set_inlets_imbibition(self, pores=None, **kwargs):
        r"""

        Parameters
        ----------
        pores : array_like
            The list of inlet pores from which the Phase can enter the Network
        """
        if 'inlets' in kwargs.keys():
            pores = kwargs['inlets']

        self.setup_imbibition()
        self._net['pore.inlet_imbibition'] = pores
        # Perform initial analysis on input pores
        order_list = self['pore.imbibition_order']
        self.Pqueue = []
        [hq.heappush(self.Pqueue, P) for P in order_list[pores]] #put the BC pores in the queue

    def find_new_filling(self, pc=[], lpc=None,# case='wetting',
                         pores=None, throats=None):
        r"""Return the elements that change their filling for the given pc.
        case = 'wetting' or 'drying_wetting'
        pc = [pc0, pc1], pc1 > pc0
        """
        if lpc is not None:
            pc = -sp.power(10, lpc)
        elif sp.size(pc)==0:
            print('Either pc or lpc must be specified')

        if sp.size(pc) != 2:
            print('The supplied pc or lpc must be a list/array with 2 entries')

        if sp.size(self.name.split('_')) == 2:
            case = 'drying_wetting'
        else:
            case = 'wetting'
        prop = case + '_inv_pc'

        if (pores is None) and (throats is None):
            ppc = self['pore.' + prop]
            tpc = self['throat.' + prop]
            ptag0 = sp.where(ppc <= pc[0])[0]
            ttag0 = sp.where(tpc <= pc[0])[0]
            ptag1 = sp.where(ppc <= pc[1])[0]
            ttag1 = sp.where(tpc <= pc[1])[0]
            ptag = sp.where(sp.in1d(ptag1, ptag0, invert=True))[0]
            ttag = sp.where(sp.in1d(ttag1, ttag0, invert=True))[0]
        return ptag1[ptag], ttag1[ttag]

    def run_wetting(self, inv_points=None, n_steps=None, n_print=1000,
                    **kwargs):
        r"""
        The default wetting process due to humid air, without accessibility and
        entrapment procedures.

        Inlet and outlet pores are not required


        Parameters
        ----------
        n_steps : int
            The number of throats to invade during this step

        inv_points: array_like,
            default: None => inv_points = all pore's and throat's pc.


        Output (Return nothing)
        -----------------------
        pore/throat.wetting_inv_seq: invasion seq
        pore/throat.wetting_inv_pc: invasion pc
        pore/throat.wetting_inv_sat: invasion saturation
        """
        self.setup_wetting(inv_points=inv_points, **kwargs)

        if n_steps is None:
            n_steps = sp.inf

        # Perform initial analysis on input pores
        Np, Nt = self.Np, self.Nt
        p_inv = -sp.ones((Np,))
        t_inv = -sp.ones((Nt,))
        p_inpc = sp.zeros(Np)
        t_inpc = sp.zeros(Nt)
        p_insat = sp.zeros(Np)
        t_insat = sp.zeros(Nt)
        p_order = self._p_order
        t_order = self._t_order
        pvol = self._net['pore.volume']
        tvol = self._net['throat.volume']
        v_total = sp.sum(pvol) + sp.sum(tvol)
        sat = 0.
        count = 0
        _tcount = self._tcount

        inv_points = list(self._wetting_pc)
        while (len(inv_points) > 0) and (count < n_steps):
#            if not sp.mod(count, n_print):
#                print('Wetting - Starting step: ', count)
            # Find pore/throat at the top of the queue
            pc = inv_points.pop(0)
            p_index = p_order[count]    # p_index is a list, might be empty
            t_index = t_order[count]

            p_inv[p_index] = _tcount    # invade
            t_inv[t_index] = _tcount    # invade

            v_liquid = sum(pvol[p_index]) + sum(tvol[t_index])
            sat += v_liquid/v_total

            p_inpc[p_index] = pc
            t_inpc[t_index] = pc
            p_insat[p_index] = sat
            t_insat[t_index] = sat

            count += 1
            _tcount += 1
        self['pore.wetting_inv_seq'] = p_inv
        self['throat.wetting_inv_seq'] = t_inv
        self['pore.wetting_inv_pc'] = p_inpc
        self['throat.wetting_inv_pc'] = t_inpc
        self['pore.wetting_inv_sat'] = p_insat
        self['throat.wetting_inv_sat'] = t_insat

    def _run_imbibition(self, n_steps=None, n_print=1000, entrapment=False,
                       **kwargs):
        r"""
        Perform scenario 2: filter the original scenario with applied ambient
            capillary pressure.

        Parameters
        ----------
        n_steps : int
            The number of invasions

        entrapment: bool
            True = Local (single-element) entrapment

        """
        if n_steps is None:
            n_steps = sp.inf

        try:
            p_queue = self.Pqueue # if set_inlets_imbibition has been performed
        except:
            self.set_inlets_imbibition(pores=self._net['pore.inlet'])
            p_queue = self.Pqueue

        t_queue = []
        p_pc = self['pore.entry_pressure']
        t_pc = self['throat.entry_pressure']
        p_sorted = self['pore.sorted']
        t_sorted = self['throat.sorted']
        p_order = self['pore.imbibition_order']
        t_order = self['throat.imbibition_order']
        p_inv = -sp.ones((self.Np,))
        t_inv = -sp.ones((self.Nt,))
        t_conns = self._net['throat.conns']
        self._imbibition_pc = []

        count = 0
        while (len(sp.r_[p_queue, t_queue]) > 0) and (count < n_steps):
#            if not sp.mod(count, n_print):
#                print('Imbibition - Starting step: ', count)
            # Find pore/throat at the top of the queue
            if p_queue:
                p = hq.heappop(p_queue)
                p_next = p_sorted[p]
                p_pc_next = p_pc[p_next]
            else:
                p_pc_next = 0

            if t_queue:
                t = hq.heappop(t_queue)
                t_next = t_sorted[t]
                t_pc_next = t_pc[t_next]
            else:
                t_pc_next = 0

            if (p_pc_next < t_pc_next):     # invade the pores, not the throats
                self._imbibition_pc.append(p_pc_next)
                self._invade_pore(p_queue, p, p_inv, p_next, p_sorted,
                                  t_queue, t_inv, t_order, entrapment)
                # put back the unused throat to the queue
                if t_pc_next:
                    hq.heappush(t_queue, t)

            elif (p_pc_next > t_pc_next):
                self._imbibition_pc.append(t_pc_next)
                self._invade_throat(t_queue, t, t_inv, t_next, t_sorted, t_conns,
                                    p_queue, p_inv, p_order, entrapment)
                # put back the unused pore to the queue
                if p_pc_next:
                    hq.heappush(p_queue, p)

            else:
                self._imbibition_pc.append(p_pc_next)
                self._invade_pore(p_queue, p, p_inv, p_next, p_sorted,
                                  t_queue, t_inv, t_order, entrapment)
                self._invade_throat(t_queue, t, t_inv, t_next, t_sorted, t_conns,
                                    p_queue, p_inv, p_order, entrapment)

            count += 1
            self._tcount += 1
        self['pore.imbibition_inv_seq'] = p_inv
        self['throat.imbibition_inv_seq'] = t_inv
        self._make_inv_pc_imbibition()

    def run_imbibition(self, inv_points=None, n_steps=None, n_print=1000,
                       entrapment=False, **kwargs):
        r"""
        Perform scenario 2: filter the original scenario with applied ambient
            capillary pressure.

        Parameters
        ----------
        n_steps : int
            The number of invasions

        entrapment: bool
            True = Local (single-element) entrapment

        """
        if inv_points is None:
            self._run_imbibition(n_steps, n_print, entrapment, **kwargs)
            return

        inv_points = sp.array(inv_points).flatten()
        inv_points.sort()

        if n_steps is None:
            n_steps = sp.inf

        try:
            p_queue = self.Pqueue # if set_inlets_imbibition has been performed
        except:
            self.set_inlets_imbibition(pores=self._net['pore.inlet'])
            p_queue = self.Pqueue

        t_queue = []
        p_pc = self['pore.entry_pressure']
        t_pc = self['throat.entry_pressure']
        p_sorted = self['pore.sorted']
        t_sorted = self['throat.sorted']
        p_order = self['pore.imbibition_order']
        t_order = self['throat.imbibition_order']
        p_inv = -sp.ones((self.Np,))
        t_inv = -sp.ones((self.Nt,))
        t_conns = self._net['throat.conns']
        self._imbibition_pc = []
#        self._imbibition_pc = inv_points

        p_ind = p_sorted[p_queue]
        p_pc_next = p_pc[p_ind]
        count = sp.searchsorted(inv_points, p_pc_next[0])
        pc = inv_points[count] + self._eps
#        self._tcount = count
        while (len(sp.r_[p_queue, t_queue]) > 0) and (count < n_steps):
#            if not sp.mod(count, n_print):
#                print('Imbibition - Starting step: ', count)
            # Find pore/throat at the top of the queue
            if p_queue:
                p_ind = p_sorted[p_queue]
                p_pc_next = p_pc[p_ind]
                p_next = p_ind[p_pc_next <= pc]
                if p_next.size > 0:
                    p_next = sp.unique(p_next)
                    for i in p_next:    #this should be replaced by arraywise
                        p = hq.heappop(p_queue)
                        self._invade_pore(p_queue, p, p_inv, p_sorted[p], p_sorted,
                                      t_queue, t_inv, t_order, entrapment)
                        self._imbibition_pc.append(p_pc[p_sorted[p]])
                        self._tcount += 1
                    t_proceed = False
            p_proceed = True

            if t_queue:
                t_ind = t_sorted[t_queue]
                t_pc_next = t_pc[t_ind]
                t_next = t_ind[t_pc_next <= pc]
                if t_next.size > 0:  #this should be replaced by arraywise
                    t_next = sp.unique(t_next)
                    for i in t_next:
                        try:
                            t = hq.heappop(t_queue)
                        except:
                            pass
                        try:
                            self._invade_throat(t_queue, t, t_inv, t_sorted[t], t_sorted, t_conns,
                                            p_queue, p_inv, p_order, entrapment)
                        except:
                            pass
                        self._imbibition_pc.append(t_pc[t_sorted[t]])
                        self._tcount += 1
                    p_proceed = False
            t_proceed = True

            if p_proceed and t_proceed:
                count += 1
#                self._tcount += 1
                pc = inv_points[count] + self._eps

        self['pore.imbibition_inv_seq'] = p_inv
        self['throat.imbibition_inv_seq'] = t_inv
        self._make_inv_pc_imbibition()

    def run_imbibition_old(self, inv_points=None, n_steps=None, n_print=1000,
                       entrapment=False, **kwargs):
        r"""
        Perform scenario 2: filter the original scenario with applied ambient
            capillary pressure.

        Parameters
        ----------
        n_steps : int
            The number of invasions

        entrapment: bool
            True = Local (single-element) entrapment

        """
        if inv_points is None:
            self._run_imbibition(n_steps, n_print, entrapment, **kwargs)
            return

        inv_points = sp.array(inv_points).flatten()
        inv_points.sort()

        if n_steps is None:
            n_steps = sp.inf

        try:
            p_queue = self.Pqueue # if set_inlets_imbibition has been performed
        except:
            self.set_inlets_imbibition(pores=self._net['pore.inlet'])
            p_queue = self.Pqueue

        t_queue = []
        p_pc = self['pore.entry_pressure']
        t_pc = self['throat.entry_pressure']
        p_sorted = self['pore.sorted']
        t_sorted = self['throat.sorted']
        p_order = self['pore.imbibition_order']
        t_order = self['throat.imbibition_order']
        p_inv = -sp.ones((self.Np,))
        t_inv = -sp.ones((self.Nt,))
        t_conns = self._net['throat.conns']
        self._imbibition_pc = inv_points

        p_ind = p_sorted[p_queue]
        p_pc_next = p_pc[p_ind]
        count = sp.searchsorted(inv_points, p_pc_next[0])
        pc = inv_points[count] + self._eps
        self._tcount = count
        while (len(sp.r_[p_queue, t_queue]) > 0) and (count < n_steps):
#            if not sp.mod(count, n_print):
#                print('Imbibition - Starting step: ', count)
            # Find pore/throat at the top of the queue

            if p_queue:
                p_ind = p_sorted[p_queue]
                p_pc_next = p_pc[p_ind]
                p_next = p_ind[p_pc_next <= pc]
                if sp.any(p_next):
                    p_next = sp.unique(p_next)
                    for i in p_next:    #this should be replaced by arraywise
                        p = hq.heappop(p_queue)
                        self._invade_pore(p_queue, p, p_inv, p_sorted[p], p_sorted,
                                      t_queue, t_inv, t_order, entrapment)
                    t_proceed = False
            p_proceed = True

            if t_queue:
                t_ind = t_sorted[t_queue]
                t_pc_next = t_pc[t_ind]
                t_next = t_ind[t_pc_next <= pc]
                if sp.any(t_next):  #this should be replaced by arraywise
                    t_next = sp.unique(t_next)
                    for i in t_next:
                        try:
                            t = hq.heappop(t_queue)
                        except:
                            pass
                        try:
                            self._invade_throat(t_queue, t, t_inv, t_sorted[t], t_sorted, t_conns,
                                            p_queue, p_inv, p_order, entrapment)
                        except:
                            pass
                    p_proceed = False
            t_proceed = True

            if p_proceed and t_proceed:
                count += 1
                self._tcount += 1
                pc = inv_points[count] + self._eps

        self['pore.imbibition_inv_seq'] = p_inv
        self['throat.imbibition_inv_seq'] = t_inv
        self._make_inv_pc_imbibition()

    def _run_drying(self, inv_site=None, n_steps=None, n_print=1000,
                    entrapment=False, case='wetting', **kwargs):
        r"""
        Perform the algorithm invasion percolation of drying

        Parameters
        ----------
        n_steps : int
            The number of throats to invaded during this step

        Revision
        ----------
        prop_name = 'drying_inv_seq'
            --> case = 'drying_wetting' or 'drying_imbibition'
            This will just change 'drying' with 'drying_wetting'

        """
        if n_steps is None:
            n_steps = sp.inf

        try:
            pores = self._inlet_drying
        except:
            pores = inv_site
            self.set_inlets_drying(pores=pores, case=case)
        case = 'drying_'+case
        p_queue = self.Pqueue

        t_queue = []
        p_pc = self['pore.entry_pressure']
        t_pc = self['throat.entry_pressure']
        p_sorted = self['pore.drying_sorted']
        t_sorted = self['throat.drying_sorted']
        p_order = self['pore.'+case+'_order']
        t_order = self['throat.'+case+'_order']
        p_inv = -sp.ones((self.Np,))
        p_inv[~self._pwet] = self._Mseq     # exclude trapped pores
        t_inv = -sp.ones((self.Nt,))
        t_inv[~self._twet] = self._Mseq     # exclude trapped pores
        t_conns = self._net['throat.conns']
        self._drying_pc = []

        count = 0
        while (len(sp.r_[p_queue, t_queue]) > 0) and (count < n_steps):
            if not sp.mod(count, n_print):
                print('Drying - Starting step: ', count)
            # Find pore/throat at the top of the queue
            if p_queue:
                p = hq.heappop(p_queue)
                p_next = p_sorted[p]
                p_pc_next = p_pc[p_next]
            else:
                p_pc_next = -1e+20

            if t_queue:
                t = hq.heappop(t_queue)
                t_next = t_sorted[t]
                t_pc_next = t_pc[t_next]
            else:
                t_pc_next = -1e+20

            if (p_pc_next > t_pc_next):     # invade the pores, not the throats
                self._drying_pc.append(p_pc_next)
                self._invade_pore(p_queue, p, p_inv, p_next, p_sorted,
                                  t_queue, t_inv, t_order, entrapment)
                # put back the unused throat to the queue
                if t_pc_next > -1e+20:
                    hq.heappush(t_queue, t)

            elif (p_pc_next < t_pc_next):
                self._drying_pc.append(t_pc_next)
                self._invade_throat(t_queue, t, t_inv, t_next, t_sorted, t_conns,
                                    p_queue, p_inv, p_order, entrapment)
                # put back the unused pore to the queue
                if p_pc_next > -1e+20:
                    hq.heappush(p_queue, p)

            else:
                self._drying_pc.append(p_pc_next)
                self._invade_pore(p_queue, p, p_inv, p_next, p_sorted,
                                  t_queue, t_inv, t_order, entrapment)
                self._invade_throat(t_queue, t, t_inv, t_next, t_sorted, t_conns,
                                    p_queue, p_inv, p_order, entrapment)

            count += 1
            self._tcount += 1
        self['pore.'+case+'_inv_seq'] = p_inv
        self['throat.'+case+'_inv_seq'] = t_inv
        self._make_inv_pc_drying(case)

    def run_drying(self, inv_points=None, inv_site=None, n_steps=None,
                    n_print=1000, entrapment=False, case='wetting', **kwargs):
        r"""
        Perform the algorithm invasion percolation of drying.

        Parameters
        ----------
        n_steps : int
            The number of elements to invade during this step

        inv_points: [None, array]
            None    perform accurate invasion by invading element one by one,
                    and record its corresponding sequence, capillary pressure,
                    and saturation values.
            array   perform coarse invasion by invading a number of elements
                    at once at the specified inv_points of capillary pressures.
                    This will record the sequence, capillary pressure and
                    saturation values as in the accurate invasion. However,
                    the order of invasion within the same inv_value may not
                    be preserved.

        Revision
        ----------
        prop_name = 'drying_inv_seq'
            --> case = 'drying_wetting' or 'drying_imbibition'
            This will just change 'drying' with 'drying_wetting'

        """
        if inv_points is None:
            self._run_drying(inv_site, n_steps, n_print, entrapment, case,
                             **kwargs)
            return

        inv_points = sp.array(inv_points).flatten()
        inv_points.sort()

        if n_steps is None:
            n_steps = sp.inf

        try:
            pores = self._inlet_drying
        except:
            pores = inv_site
            self.set_inlets_drying(pores=pores, case=case)
        case = 'drying_'+case
        p_queue = self.Pqueue

        t_queue = []
        p_pc = self['pore.entry_pressure']
        t_pc = self['throat.entry_pressure']
        p_sorted = self['pore.drying_sorted']
        t_sorted = self['throat.drying_sorted']
        p_order = self['pore.'+case+'_order']
        t_order = self['throat.'+case+'_order']
        p_inv = -sp.ones((self.Np,))
        p_inv[~self._pwet] = self._Mseq     # exclude trapped pores
        t_inv = -sp.ones((self.Nt,))
        t_inv[~self._twet] = self._Mseq     # exclude trapped pores
        t_conns = self._net['throat.conns']
        self._drying_pc = []

        p_ind = p_sorted[p_queue]
        p_pc_next = p_pc[p_ind]
        count = sp.searchsorted(inv_points, p_pc_next[0]) # searchsorted only works with ascending order
        inv_points[::-1].sort()     # descending order
        count = inv_points.size - count
        pc = inv_points[count] - self._eps
        while (len(sp.r_[p_queue, t_queue]) > 0) and (count < n_steps):
#            if not sp.mod(count, n_print):
#                print('Imbibition - Starting step: ', count)
            # Find pore/throat at the top of the queue

            if p_queue:
                p_ind = p_sorted[p_queue]
                p_pc_next = p_pc[p_ind]
                p_next = p_ind[p_pc_next > pc]
                if sp.any(p_next):
                    p_next = sp.unique(p_next)
                    for i in p_next:    #this should be replaced by arraywise
                        p = hq.heappop(p_queue)
                        self._invade_pore(p_queue, p, p_inv, p_sorted[p], p_sorted,
                                      t_queue, t_inv, t_order, entrapment)
                        self._tcount += 1
                        self._drying_pc.append(p_pc[p_sorted[p]])
                    t_proceed = False
            p_proceed = True

            if t_queue:
                t_ind = t_sorted[t_queue]
                t_pc_next = t_pc[t_ind]
                t_next = t_ind[t_pc_next > pc]
                if sp.any(t_next):  #this should be replaced by arraywise
                    t_next = sp.unique(t_next)
                    for i in t_next:
                        try:
                            t = hq.heappop(t_queue)
                        except:
                            pass
                        try:
                            self._invade_throat(t_queue, t, t_inv, t_sorted[t], t_sorted, t_conns,
                                            p_queue, p_inv, p_order, entrapment)
                        except:
                            pass
                        self._tcount += 1
                        self._drying_pc.append(t_pc[t_sorted[t]])
                    p_proceed = False
            t_proceed = True

            if p_proceed and t_proceed:
                count += 1
                pc = inv_points[count] - self._eps

        self['pore.'+case+'_inv_seq'] = p_inv
        self['throat.'+case+'_inv_seq'] = t_inv
        self._make_inv_pc_drying(case)

    def run_drying_worked(self, inv_points=None, inv_site=None, n_steps=None,
                    n_print=1000, entrapment=False, case='wetting', **kwargs):
        r"""
        Perform the algorithm invasion percolation of drying

        Parameters
        ----------
        n_steps : int
            The number of throats to invaded during this step

        Revision
        ----------
        prop_name = 'drying_inv_seq'
            --> case = 'drying_wetting' or 'drying_imbibition'
            This will just change 'drying' with 'drying_wetting'

        """
        if inv_points is None:
            self._run_drying(inv_site, n_steps, n_print, entrapment, case,
                             **kwargs)
            return

        inv_points = sp.array(inv_points).flatten()
        inv_points.sort()

        if n_steps is None:
            n_steps = sp.inf

        try:
            pores = self._inlet_drying
        except:
            pores = inv_site
            self.set_inlets_drying(pores=pores, case=case)
        case = 'drying_'+case
        p_queue = self.Pqueue

        t_queue = []
        p_pc = self['pore.entry_pressure']
        t_pc = self['throat.entry_pressure']
        p_sorted = self['pore.drying_sorted']
        t_sorted = self['throat.drying_sorted']
        p_order = self['pore.'+case+'_order']
        t_order = self['throat.'+case+'_order']
        p_inv = -sp.ones((self.Np,))
        p_inv[~self._pwet] = self._Mseq     # exclude trapped pores
        t_inv = -sp.ones((self.Nt,))
        t_inv[~self._twet] = self._Mseq     # exclude trapped pores
        t_conns = self._net['throat.conns']
        self._drying_pc = inv_points

        p_ind = p_sorted[p_queue]
        p_pc_next = p_pc[p_ind]
        count = sp.searchsorted(inv_points, p_pc_next[0]) # searchsorted only works with ascending order
        inv_points[::-1].sort()     # descending order
        count = inv_points.size - count
        pc = inv_points[count] - self._eps
        self._tcount = count
        while (len(sp.r_[p_queue, t_queue]) > 0) and (count < n_steps):
#            if not sp.mod(count, n_print):
#                print('Imbibition - Starting step: ', count)
            # Find pore/throat at the top of the queue

            if p_queue:
                p_ind = p_sorted[p_queue]
                p_pc_next = p_pc[p_ind]
                p_next = p_ind[p_pc_next > pc]
                if sp.any(p_next):
                    p_next = sp.unique(p_next)
                    for i in p_next:    #this should be replaced by arraywise
                        p = hq.heappop(p_queue)
                        self._invade_pore(p_queue, p, p_inv, p_sorted[p], p_sorted,
                                      t_queue, t_inv, t_order, entrapment)
                    t_proceed = False
            p_proceed = True

            if t_queue:
                t_ind = t_sorted[t_queue]
                t_pc_next = t_pc[t_ind]
                t_next = t_ind[t_pc_next > pc]
                if sp.any(t_next):  #this should be replaced by arraywise
                    t_next = sp.unique(t_next)
                    for i in t_next:
                        try:
                            t = hq.heappop(t_queue)
                        except:
                            pass
                        try:
                            self._invade_throat(t_queue, t, t_inv, t_sorted[t], t_sorted, t_conns,
                                            p_queue, p_inv, p_order, entrapment)
                        except:
                            pass
                    p_proceed = False
            t_proceed = True

            if p_proceed and t_proceed:
                count += 1
                self._tcount += 1
                pc = inv_points[count] - self._eps

        self['pore.'+case+'_inv_seq'] = p_inv
        self['throat.'+case+'_inv_seq'] = t_inv
        self._make_inv_pc_drying(case)

    def _make_inv_pc_imbibition(self):
        r'''Create imbibition properties:
        pore/throat.inv_pc
        pore/throat.inv_sat
        self._imbibition_inv_pc ~ self._imbibition_inv_sat
        '''
        pseq = self['pore.imbibition_inv_seq']
        tseq = self['throat.imbibition_inv_seq']
        Np, Nt = self.Np, self.Nt
        psat = sp.zeros(Np)
        tsat = sp.zeros(Nt)
        ppc = sp.zeros(Np)
        tpc = sp.zeros(Nt)
        # initial imbibition_inv_pc
        ipc = self._imbibition_pc
        inv_pc = []
        inv_pc.append(ipc[0])
        inv_pores = pseq==0
        inv_throats = tseq==0
        ppc[inv_pores] = inv_pc[-1]
        # initial sat, i=0
        inv_sat = []
        pvol = self._net['pore.volume']
        tvol = self._net['throat.volume']
        v_total = sp.sum(pvol) + sp.sum(tvol)
        v_liquid = (sum(pvol[inv_pores]) + sum(tvol[inv_throats]))
        sat = v_liquid/v_total
        inv_sat.append(sat)
        psat[inv_pores] = sat
        tsat[inv_throats] = sat
        for i, pc in enumerate(ipc[1:]):
            j = i+1     # j=1:
            inv_pores = pseq ==j
            inv_throats = tseq ==j
            # sat
            v_liquid = (sum(pvol[inv_pores]) + sum(tvol[inv_throats]))
            sat += v_liquid/v_total
            psat[inv_pores] = sat
            tsat[inv_throats] = sat
            # record pc for w(pc) s.t. pc is strictly monotonously increasing
            if pc > inv_pc[-1]:
                inv_pc.append(pc)
                inv_sat.append(sat)
            elif j==len(ipc)-1:
                inv_sat.append(sat)
            ppc[inv_pores] = inv_pc[-1]
            tpc[inv_throats] = inv_pc[-1]
        # inv_sat corresponds to the pc values in inv_pc:
        # sat[pc] = inv_sat[inv_pc]
        self._imbibition_inv_pc = inv_pc
        self._imbibition_inv_sat = inv_sat
        self['pore.imbibition_inv_sat'] = psat
        self['throat.imbibition_inv_sat'] = tsat
        self['pore.imbibition_inv_pc'] = ppc
        self['throat.imbibition_inv_pc'] = tpc

    def _make_inv_pc_drying(self, case):
        r'''Create drying properties:
        pore/throat.inv_pc
        pore/throat.inv_sat
        self._imbibition_inv_pc ~ self._imbibition_inv_sat
        '''

        pseq = self['pore.'+case+'_inv_seq']
        tseq = self['throat.'+case+'_inv_seq']
        Np, Nt = self.Np, self.Nt
        psat = 10*sp.ones(Np)
        tsat = 10*sp.ones(Nt)
        ppc = sp.zeros(Np)
        tpc = sp.zeros(Nt)
        # initial drying_inv_pc
        ipc = self._drying_pc   # limited to invaded pores only, see run_drying
        inv_pc = []
        # initial condition
#        inv_pc.append(sp.amax(sp.r_[self._p_inv_pc, self._t_inv_pc]))
#        inv_pores = pseq<0
#        inv_throats = tseq<0
#        ppc[inv_pores] = inv_pc[-1]
#        tpc[inv_throats] = inv_pc[-1]
        # iterative invasion
        inv_pc.append(ipc[0])
        inv_pores = pseq==0
        inv_throats = tseq==0
        ppc[inv_pores] = inv_pc[-1]
        tpc[inv_throats] = inv_pc[-1]
        # initial sat, i=0
        inv_sat = []
        pvol = self._net['pore.volume']
        tvol = self._net['throat.volume']
        v_total = sp.sum(pvol) + sp.sum(tvol)
        # initial condition
        v_liquid = (sum(pvol[self._pwet]) + sum(tvol[self._twet]))
        sat0 = v_liquid/v_total
#        inv_sat.append(sat0)
        # iterative invasion
        v_liquid = (sum(pvol[inv_pores]) + sum(tvol[inv_throats]))
        v_liquid += (sum(pvol[~self._pwet]) + sum(tvol[~self._twet]))
        sat = v_liquid/v_total
        inv_sat.append(sat0-sat)
        psat[inv_pores] = inv_sat[-1]
        tsat[inv_throats] = inv_sat[-1]
        for i, pc in enumerate(ipc[1:]):
            j = i+1     # j=1:
            inv_pores = pseq == j
            inv_throats = tseq == j
            # sat
            v_liquid += (sum(pvol[inv_pores]) + sum(tvol[inv_throats]))
            sat = v_liquid/v_total
            psat[inv_pores] = 1-sat
            tsat[inv_throats] = 1-sat
            # record pc for w(pc) s.t. pc is strictly monotonously decreasing
            if pc < inv_pc[-1]:
                inv_pc.append(pc)
                inv_sat.append(1-sat)
            elif j==len(ipc)-1:
                inv_sat.append(1-sat)
            ppc[inv_pores] = inv_pc[-1]
            tpc[inv_throats] = inv_pc[-1]
        # inv_sat corresponds to the pc values in inv_pc:
        # sat[pc] = inv_sat[inv_pc]
        self._drying_inv_pc = inv_pc
        self._drying_inv_sat = inv_sat
        self['pore.'+case+'_inv_sat'] = psat
        self['throat.'+case+'_inv_sat'] = tsat
        self['pore.'+case+'_inv_pc'] = ppc
        self['throat.'+case+'_inv_pc'] = tpc

    def _invade_pore(self, p_queue, p, p_inv, p_next, p_sorted,
                     t_queue, t_inv, t_order, entrapment=False):
        if entrapment:  # what is this part for? I dont remember. Must be for a special case
            p_counter = counter(p_queue)    # count element repetition
            n_duplicate = p_counter[p]
            n_conns = self._net.num_neighbors(p_next)
            if n_duplicate and n_duplicate == n_conns-1:  # pore is trapped
                if not self._net['pore.inlet'][p_next]:   # except inlet pores
                    # remove this pore and all duplicates from queue
                    self._del_duplicate(p_queue, p)
                    return    # do not invade, just let it pop out

        p_inv[p_next] = self._tcount    # invade
        # add connected throats to queue
        self._queue_throat(p_next, t_queue, t_inv, t_order)
        # If pore is duplicated
        self._del_duplicate(p_queue, p)

    def _queue_throat(self, p_next, t_queue, t_inv, t_order):
        r'''Add connected throats of invaded pore to throat queue
        '''
        # Find throats connected to newly invaded pore
        Ts = self._net.find_neighbor_throats(pores=p_next)
        try:
            Ts = Ts[t_inv[Ts] < 0]  # Remove invaded & trapped throats from Ts
            [hq.heappush(t_queue, T) for T in t_order[Ts]]  # addthroat toqueue
        except:
            pass    # print('single surface pore/cluster: ', p_next)

    def _invade_throat(self, t_queue, t, t_inv, t_next, t_sorted, t_conns,
                       p_queue, p_inv, p_order, entrapment=False):
        if entrapment:
            t_counter = counter(t_queue)
            n_duplicate = t_counter[t]
            if n_duplicate:
                return    # do nothing just let it pop out

        t_inv[t_next] = self._tcount
        # Add connected pores of invaded throat to pore queue
        self._queue_pore(t_next, t_conns, p_queue, p_inv, p_order)
        # If throat is duplicated
        self._del_duplicate(t_queue, t)

    def _queue_pore(self, t_next, t_conns, p_queue, p_inv, p_order):
        r'''Add connected pores of invaded throat to pore queue
        '''
        # Find pores connected to newly invaded throat
        Ps = t_conns[t_next]
        try:
            Ps = Ps[p_inv[Ps] < 0]  # Remove invaded pores from Ps
            [hq.heappush(p_queue, P) for P in p_order[Ps]]  # addpore toqueue
        except:
            pass    # print('single surface pore/cluster: ', t_next)

    def _del_duplicate(self, queue, entries):
        # If throat is duplicated
        try:
            while len(queue) > 0 and queue[0] == entries:
                    # Note: Preventing duplicate entries below might save some
                    entry = hq.heappop(queue)
        except:
            entries = sp.array(list(entries))
            for entry in entries:
                while len(queue) > 0 and queue[0] == entry:
                    # Note: Preventing duplicate entries below might save some
                    entry = hq.heappop(queue)


    def evaluate_trapping_wetting(self, p_outlets, mode='clone'):
        r"""
        Finds trapped pores and throats after a full wetting
        percolation simulation has been run.
        Given moisture distribution, evaluate which dry pores
        are trapped/isolated (having no air path to outlet)
        at respective invasion value.
        So, it's not evaluating the to-be-wet elements!

        Parameters
        ----------
        p_outlets : array_like
            A list of pores that define the wetting phase outlets.
            Disconnection from these outlets results in trapping.

        prop_name: string
            default: 'clone' => create new properties with 'trapping' ending.
            other: 'replace' => replacing the result from run_wetting.

        Returns
    
        It creates arrays called ``pore.trapped`` and ``throat.trapped``, but
        also adjusts the ``pore.prop_name`` and ``throat.prop_name`` arrays to set
        trapped locations to have zero invasion pressure.
        """
        inv_points = list(self._wetting_pc)
        ptrappc = sp.zeros([self.Np, ], dtype=float)
        ttrappc = sp.zeros([self.Nt, ], dtype=float)
        ptrapseq = sp.ones([self.Np, ], dtype=int)*self._Mseq
        ttrapseq = sp.ones([self.Nt, ], dtype=int)*self._Mseq
        Psat = sp.copy(self['pore.wetting_inv_sat'])
        Tsat = sp.copy(self['throat.wetting_inv_sat'])
        ppc = self['pore.wetting_inv_pc']
        tpc = self['throat.wetting_inv_pc']
        pseq = self['pore.wetting_inv_seq']
        tseq = self['throat.wetting_inv_seq']
        pvol = self._net['pore.volume']
        tvol = self._net['throat.volume']
        vol_total = sp.sum(pvol) + sp.sum(tvol)
        ptrap = sp.asarray([False] * self.Np)
        ttrap = sp.asarray([False] * self.Nt)
        net = self._net
        conns = net['throat.conns']
        for i, inv_val in enumerate(inv_points):
            # Find clusters of defender pores
            Pinvaded =  ppc <= inv_val
            Tinvaded = tpc <= inv_val
            # 0 = all open, 1=1 pore filled,
            # 2=2 pores filled 3=2 pores + 1 throat filled
            Cstate = sp.sum(Pinvaded[conns], axis=1) + Tinvaded
            # pore clusters (Np long) determined by uninvaded throats
            clusters = net.find_clusters(Cstate == 0)
            # Clean up clusters (invaded = -1, defended >=0)
            clusters = clusters * (~Pinvaded) - (Pinvaded)
            # Identify cluster numbers connected to outlet sites
            out_clusters = sp.unique(clusters[p_outlets])
            #pore clusters not connected to outlet
            trapped_pores = ~sp.in1d(clusters, out_clusters)
            trapped_pores[Pinvaded] = False
            # Maybe there are trapped throats (having 2 invaded pores)
            # except invaded throats
            Tinds = (ttrappc == 0) * (Cstate == 2)
            Tinds[sp.logical_and(Tinds, Tinvaded)] = False
            if sp.any(trapped_pores):
                # Pinds: dry pores becoming trapped at this inv point
                # Tinds: dry throats becoming trapped at this inv point
                # both are not to-be-wet elements! so, they dont reduce sat at
                # this step, but the previous-step trapped elements do!
                # These previous trapped elements are specified by p/tcurrent.
                # Pinds \in ~Pinvaded, pcurrent \in Pinvaded
                # sp.where(trappedpores) and still zero in pore.trapped
                Pinds = (ptrappc == 0) * trapped_pores
                ptrappc[Pinds] = inv_val
                ptrapseq[Pinds] = i
                # Trapped throats:
                trapped_throat_array = sp.asarray([False] * len(Cstate))
                # 1. all neighbor throats of a trapped pore
                trapped_throats = net.find_neighbor_throats(trapped_pores)
                if sp.any(trapped_throats):
                    trapped_throat_array[trapped_throats] = True
                Tinds = (ttrappc == 0) * trapped_throat_array
                # 2. all throats with 2 wet pores (Cstate==2)
                Tinds += (ttrappc == 0) * (Cstate == 2)
                # except invaded throats
                Tinds[sp.logical_and(Tinds, Tinvaded)] = False
                ttrappc[Tinds] = inv_val
                ttrapseq[Tinds] = i
                # Update list of trapped pores & throats
                ptrap[Pinds] = True
                ttrap[Tinds] = True

            elif sp.any(Tinds):
                ttrappc[Tinds] = inv_val
                ttrapseq[Tinds] = i
                ttrap[Tinds] = True
            # Correction to saturation: inv_sat
            pnow = (pseq==i)*ptrap
            tnow = (tseq==i)*ttrap
            vol_trapped = sp.sum(pvol[pnow]) + sp.sum(tvol[tnow])
            sat = vol_trapped/vol_total

            # revise other higher saturations
            pindsat = pnow + ~Pinvaded
            tindsat = tnow + ~Tinvaded
            Psat[pindsat] = Psat[pindsat] - (sat*sp.any(pindsat))
            Tsat[tindsat] = Tsat[tindsat] - (sat*sp.any(tindsat))

        # modify results from run_wetting with out-of-range values.
        Psat[ptrap] = 10
        Tsat[ttrap] = 10
        self['pore.wetting_trapped_pc'] = ptrappc
        self['throat.wetting_trapped_pc'] = ttrappc
        self['pore.wetting_trapped_seq'] = ptrapseq
        self['throat.wetting_trapped_seq'] = ttrapseq
        if mode=='replace':
            self['pore.wetting_inv_pc'][ptrap] = 0
            self['throat.wetting_inv_pc'][ttrap] = 0
            self['pore.wetting_inv_seq'][ptrap] = self._Mseq
            self['throat.wetting_inv_seq'][ttrap] = self._Mseq
            self['pore.wetting_inv_sat'] = Psat
            self['throat.wetting_inv_sat'] = Tsat

        elif mode=='clone':
            self['pore.wetting_inv_pc_trapping'] = sp.copy(self['pore.wetting_inv_pc'])
            self['throat.wetting_inv_pc_trapping'] = sp.copy(self['throat.wetting_inv_pc'])
            self['pore.wetting_inv_pc_trapping'][ptrap] = 0.
            self['throat.wetting_inv_pc_trapping'][ttrap] = 0.
            self['pore.wetting_inv_seq_trapping'] = sp.copy(self['pore.wetting_inv_seq'])
            self['throat.wetting_inv_seq_trapping'] = sp.copy(self['throat.wetting_inv_seq'])
            self['pore.wetting_inv_seq_trapping'][ptrap] = self._Mseq
            self['throat.wetting_inv_seq_trapping'][ttrap] = self._Mseq
            self['pore.wetting_inv_sat_trapping'] = Psat
            self['throat.wetting_inv_sat_trapping'] = Tsat
        else:
            raise Exception('Mode argument is either \'replace\' or \'clone\'')
        
        '''
        np.savetxt("wetting_OLD_CB/trapped_pores.txt", np.where(ptrap)[0], fmt="%s")
        np.savetxt("wetting_OLD_CB/trapped_throats.txt", np.where(ttrap)[0], fmt="%s")
        np.savetxt("wetting_OLD_CB/pressure_pores.txt", self['pore.wetting_inv_pc_trapping'], fmt="%s")
        np.savetxt("wetting_OLD_CB/pressure_throats.txt", self['throat.wetting_inv_pc_trapping'], fmt="%s")
        np.savetxt("wetting_OLD_CB/inv_seq_pores.txt", self['pore.wetting_inv_seq_trapping'], fmt="%s")
        np.savetxt("wetting_OLD_CB/inv_seq_throat.txt", self['throat.wetting_inv_seq_trapping'], fmt="%s")
        np.savetxt("wetting_OLD_CB/sat_pores.txt", self['pore.wetting_inv_sat_trapping'], fmt="%s")
        np.savetxt("wetting_OLD_CB/sat_throats.txt", self['throat.wetting_inv_sat_trapping'], fmt="%s")
        np.savetxt("wetting_OLD_CB/ptrap_pc.txt", ptrappc, fmt="%s")
        np.savetxt("wetting_OLD_CB/ttrap_pc.txt", ttrappc, fmt="%s")
        np.savetxt("wetting_OLD_CB/pore_volumes_trap.txt", pvol[np.where(ptrap)[0]], fmt="%s")
        np.savetxt("wetting_OLD_CB/throat_volumes_trap.txt", tvol[np.where(ptrap)[0]], fmt="%s")
        '''
        '''
        with open("wetting_OLD_CB/some_values.txt", "w") as f:
            f.write(f"Number of trapped pores: {len(np.where(ptrap)[0])}\n")
            f.write(f"Number of trapped throats: {len(np.where(ttrap)[0])}\n")
            f.write(f"min(Psat trap): {min(Psat_trap)}\n")
        '''
        print("Number of trapped pores:", len(np.where(ptrap)[0]))
        print("Number of trapped throats:", len(np.where(ttrap)[0]))
        print("Number of trapped pores:", len(np.where(ptrap)[0]))
        print("Number of trapped throats:", len(np.where(ttrap)[0]))
 
    def evaluate_trapping_wetting_fast(self, p_outlets, mode='clone'):
        """
        Added on 27/02/2026 by Homam.

        Masson-style reverse-uninvasion using a pore+throat bipartite CSR graph.
        Produces the same outputs as `evaluate_trapping_wetting.
        """
        start = time.perf_counter()
        inv_points = list(self._wetting_pc)  
        pvol = self._net['pore.volume']  
        tvol = self._net['throat.volume']
        vol = np.concatenate((pvol, tvol))
        vol_mat = np.sum(pvol) + np.sum(tvol)

        Np = self.Np
        Nt = self.Nt
        N_tot = Np + Nt

        ptrappc = np.zeros(Np)
        ttrappc = np.zeros(Nt)
        
        p_pc = self['pore.entry_pressure']
        t_pc = self['throat.entry_pressure']

        pseq = self['pore.wetting_inv_seq']
        tseq = self['throat.wetting_inv_seq']
        tot_seq = np.concatenate((pseq, tseq))
        Mseq = self._Mseq
        pseqtrap = Mseq*np.ones([Np, ], dtype=float)
        tseqtrap = Mseq*np.ones([Nt, ], dtype=float)
        conns = self._net['throat.conns']  # shape (Nt, 2) with pore indices


        # build pore–throat bipartite CSR adjacency for all N_tot nodes
        AM = lil_matrix((N_tot, N_tot), dtype=np.int8) 
        for ti, (r, c) in enumerate(conns):
            e = Np + ti
            AM[r, e] = 1; AM[e, r] = 1
            AM[c, e] = 1; AM[e, c] = 1
        AM = AM.tocsr()
        indices = AM.indices
        indptr = AM.indptr

        po_index = np.asarray(np.where(p_outlets)[0], dtype=np.int64)

        (
        trapped_elements, 
        trapped_steps,  
        vol_trapped,
        pore_id
        ) = _find_trapped_pores(tot_seq, indices, indptr, po_index, vol)
        
        '''
        trapped_elements: Indices of the trapped pores.
        trapped_steps: Steps at which pores are trapped.
        vol_trapped: The sum of total trapped pores' volume.
        pore_id: The pore indices in the order they are uninvaded.

        Note: For the calculation of the moisture content, by summing up the total trapped volume, we              
              have the final value of the volumes filled with moisture, and from the last step to the              
              first, for each pore which is untrapped we will substract it's volume from the total v-
              olume at that step, which results in a step by step moisture content.
        '''

        vol_liq = vol_mat - vol_trapped
        sat = vol_liq / vol_mat
        sat_array = np.zeros(N_tot)
        vol[trapped_elements] = 0

        for ind in pore_id:
            sat_array[ind] = sat
            sat = sat - (vol[ind]/vol_mat)
            
        Psat_trap = sat_array[:Np]
        Tsat_trap = sat_array[Np:]

        ptrap = trapped_elements[:Np]
        ttrap = trapped_elements[Np:]
        
        end = time.perf_counter() 
        print("Time taken for calculation of the trapped pores:", end-start)

        # write primary trapping outputs
        Psat_trap[ptrap] = 10
        Tsat_trap[ttrap] = 10

        ptrap_ind = np.where(ptrap)[0]
        ttrap_ind = np.where(ttrap)[0]
        
        ptrappc[ptrap_ind] = p_pc[ptrap_ind]
        ttrappc[ttrap_ind] = t_pc[ttrap_ind]

        self['pore.wetting_trapped_pc'] = ptrappc
        self['throat.wetting_trapped_pc'] = ttrappc

        # apply clone/replace semantics exactly like legacy
        if mode == 'replace':
            self['pore.wetting_inv_pc'][ptrap] = 0
            self['throat.wetting_inv_pc'][ttrap] = 0
            self['pore.wetting_inv_seq'][ptrap] = Mseq
            self['throat.wetting_inv_seq'][ttrap] = Mseq
            self['pore.wetting_inv_sat'] = Psat_trap
            self['throat.wetting_inv_sat'] = Tsat_trap
        elif mode == 'clone':
            self['pore.wetting_inv_pc_trapping'] = np.copy(self['pore.wetting_inv_pc'])
            self['throat.wetting_inv_pc_trapping'] = np.copy(self['throat.wetting_inv_pc'])
            self['pore.wetting_inv_pc_trapping'][ptrap] = 0.0
            self['throat.wetting_inv_pc_trapping'][ttrap] = 0.0
            self['pore.wetting_inv_seq_trapping'] = np.copy(self['pore.wetting_inv_seq'])
            self['throat.wetting_inv_seq_trapping'] = np.copy(self['throat.wetting_inv_seq'])
            self['pore.wetting_inv_seq_trapping'][ptrap] = Mseq
            self['throat.wetting_inv_seq_trapping'][ttrap] = Mseq
            self['pore.wetting_inv_sat_trapping'] = Psat_trap
            self['throat.wetting_inv_sat_trapping'] = Tsat_trap
        else:
            raise Exception("Mode argument must be 'replace' or 'clone'")
        
        print("______Wetting Masson's Aglorithm______")
        print("min(Psat trap):", min(Psat_trap))
        print("max(Psat trap):", max(Psat_trap))
        print("min(Tsat trap):", min(Tsat_trap))
        print("max(Tsat trap):", max(Tsat_trap))
        print("Number of trapped pores for wetting:", len(np.where(ptrap)[0]))
        print("Number of trapped throats for wetting:", len(np.where(ttrap)[0]))
        
        '''
        np.savetxt("wetting_NEW_CB/trapped_pores.txt", np.where(ptrap)[0], fmt="%s")
        np.savetxt("wetting_NEW_CB/trapped_throats.txt", np.where(ttrap)[0], fmt="%s")
        np.savetxt("wetting_NEW_CB/pressure_pores.txt", self['pore.wetting_inv_pc_trapping'], fmt="%s")
        np.savetxt("wetting_NEW_CB/pressure_throats.txt", self['throat.wetting_inv_pc_trapping'], fmt="%s")
        np.savetxt("wetting_NEW_CB/inv_seq_pores.txt", self['pore.wetting_inv_seq_trapping'], fmt="%s")
        np.savetxt("wetting_NEW_CB/inv_seq_throat.txt", self['throat.wetting_inv_seq_trapping'], fmt="%s")
        np.savetxt("wetting_NEW_CB/sat_pores.txt", self['pore.wetting_inv_sat_trapping'], fmt="%s")
        np.savetxt("wetting_NEW_CB/sat_throats.txt", self['throat.wetting_inv_sat_trapping'], fmt="%s")
        np.savetxt("wetting_NEW_CB/ptrap_pc.txt", ptrappc, fmt="%s")
        np.savetxt("wetting_NEW_CB/ttrap_pc.txt", ttrappc, fmt="%s")
        np.savetxt("wetting_NEW_CB/pore_volumes_trap.txt", pvol[np.where(ptrap)[0]], fmt="%s")
        np.savetxt("wetting_NEW_CB/throat_volumes_trap.txt", tvol[np.where(ptrap)[0]], fmt="%s")

        with open("wetting_NEW_CB/some_values.txt", "w") as f:
            f.write(f"Number of trapped pores: {len(np.where(ptrap)[0])}\n")
            f.write(f"Number of trapped throats: {len(np.where(ttrap)[0])}\n")
            f.write(f"trapped vol: {vol_trapped}\n")
            f.write(f"total vol: {vol_mat}\n")
            f.write(f"liquid vol: {vol_liq}\n")
            f.write(f"min(Psat trap): {min(Psat_trap)}\n")
        '''
        print("Number of trapped pores:", len(np.where(ptrap)[0]))
        print("Number of trapped throats:", len(np.where(ttrap)[0]))
   
    def evaluate_trapping_imbibition_fast(self, p_outlets, mode='clone'):
        """
        Added on 21/11/2025 by Homam.

        Masson-style reverse-uninvasion using a pore+throat bipartite CSR graph.
        Produces the same outputs as `evaluate_trapping_imbibition.
        """
        start = time.perf_counter()
        inv_points = list(self._imbibition_pc)  
        pvol = self._net['pore.volume']  
        tvol = self._net['throat.volume']
        vol = np.concatenate((pvol, tvol))
        vol_mat = np.sum(pvol) + np.sum(tvol)

        Np = self.Np
        Nt = self.Nt
        N_tot = Np + Nt

        ptrappc = np.zeros(Np)
        ttrappc = np.zeros(Nt)
        
        p_pc = self['pore.entry_pressure']
        t_pc = self['throat.entry_pressure']

        pseq = self['pore.imbibition_inv_seq']
        tseq = self['throat.imbibition_inv_seq']
        Mseq = self._Mseq
        pseqtrap = Mseq*np.ones([Np, ], dtype=float)
        tseqtrap = Mseq*np.ones([Nt, ], dtype=float)
        conns = self._net['throat.conns']  # shape (Nt, 2) with pore indices


        # build pore–throat bipartite CSR adjacency for all N_tot nodes
        AM = lil_matrix((N_tot, N_tot), dtype=np.int8) 
        for ti, (r, c) in enumerate(conns):
            e = Np + ti
            AM[r, e] = 1; AM[e, r] = 1
            AM[c, e] = 1; AM[e, c] = 1
        AM = AM.tocsr()
        indices = AM.indices
        indptr = AM.indptr

        po_index = np.asarray(np.where(p_outlets)[0], dtype=np.int64)

        pseq = self['pore.imbibition_inv_seq']
        tseq = self['throat.imbibition_inv_seq']
        tot_seq = np.concatenate((pseq, tseq))

        (
        trapped_elements, 
        trapped_steps,  
        vol_trapped,
        pore_id
        ) = _find_trapped_pores(tot_seq, indices, indptr, po_index, vol)
        
        '''
        trapped_elements: Indices of the trapped pores.
        trapped_steps: Steps at which pores are trapped.
        vol_trapped: The sum of total trapped pores' volume.
        pore_id: The pore indices in the order they are uninvaded.

        Note: For the calculation of the moisture content, by summing up the total trapped volume, we              
              have the final value of the volumes filled with moisture, and from the last step to the              
              first, for each pore which is untrapped we will substract it's volume from the total v-
              olume at that step, which results in a step by step moisture content.
        '''

        vol_liq = vol_mat - vol_trapped
        sat = vol_liq / vol_mat
        sat_array = np.zeros(N_tot)
        vol[trapped_elements] = 0

        for ind in pore_id:
            sat_array[ind] = sat
            sat = sat - (vol[ind]/vol_mat)
            
        Psat_trap = sat_array[:Np]
        Tsat_trap = sat_array[Np:]

        ptrap = trapped_elements[:Np]
        ttrap = trapped_elements[Np:]
        
        end = time.perf_counter() 
        print("Time taken for calculation of the trapped pores:", end-start)

        # write primary trapping outputs
        Psat_trap[ptrap] = 10
        Tsat_trap[ttrap] = 10

        ptrap_ind = np.where(ptrap)[0]
        ttrap_ind = np.where(ttrap)[0]
        
        ptrappc[ptrap_ind] = p_pc[ptrap_ind]
        ttrappc[ttrap_ind] = t_pc[ttrap_ind]

        self['pore.imbibition_trapped_pc'] = ptrappc
        self['throat.imbibition_trapped_pc'] = ttrappc

        # apply clone/replace semantics exactly like legacy
        if mode == 'replace':
            self['pore.imbibition_inv_pc'][ptrap] = 0
            self['throat.imbibition_inv_pc'][ttrap] = 0
            self['pore.imbibition_inv_seq'][ptrap] = Mseq
            self['throat.imbibition_inv_seq'][ttrap] = Mseq
            self['pore.imbibition_inv_sat'] = Psat_trap
            self['throat.imbibition_inv_sat'] = Tsat_trap
        elif mode == 'clone':
            self['pore.imbibition_inv_pc_trapping'] = np.copy(self['pore.imbibition_inv_pc'])
            self['throat.imbibition_inv_pc_trapping'] = np.copy(self['throat.imbibition_inv_pc'])
            self['pore.imbibition_inv_pc_trapping'][ptrap] = 0.0
            self['throat.imbibition_inv_pc_trapping'][ttrap] = 0.0
            self['pore.imbibition_inv_seq_trapping'] = np.copy(self['pore.imbibition_inv_seq'])
            self['throat.imbibition_inv_seq_trapping'] = np.copy(self['throat.imbibition_inv_seq'])
            self['pore.imbibition_inv_seq_trapping'][ptrap] = Mseq
            self['throat.imbibition_inv_seq_trapping'][ttrap] = Mseq
            self['pore.imbibition_inv_sat_trapping'] = Psat_trap
            self['throat.imbibition_inv_sat_trapping'] = Tsat_trap
        else:
            raise Exception("Mode argument must be 'replace' or 'clone'")
        
        print("______Masson's Aglorithm______")
        print("min(Psat trap):", min(Psat_trap))
        print("max(Psat trap):", max(Psat_trap))
        print("min(Tsat trap):", min(Tsat_trap))
        print("max(Tsat trap):", max(Tsat_trap))
        print("Number of trapped pores:", len(np.where(ptrap)[0]))
        print("Number of trapped throats:", len(np.where(ttrap)[0]))
        # print("Trapped Pores:",ptrap)
        # print("Trapped Throats:",ttrap)
        
        '''
        np.savetxt("NEW_CB/trapped_pores.txt", np.where(ptrap)[0], fmt="%s")
        np.savetxt("NEW_CB/trapped_throats.txt", np.where(ttrap)[0], fmt="%s")
        np.savetxt("NEW_CB/pressure_pores.txt", self['pore.imbibition_inv_pc_trapping'], fmt="%s")
        np.savetxt("NEW_CB/pressure_throats.txt", self['throat.imbibition_inv_pc_trapping'], fmt="%s")
        np.savetxt("NEW_CB/inv_seq_pores.txt", self['pore.imbibition_inv_seq_trapping'], fmt="%s")
        np.savetxt("NEW_CB/inv_seq_throat.txt", self['throat.imbibition_inv_seq_trapping'], fmt="%s")
        np.savetxt("NEW_CB/sat_pores.txt", self['pore.imbibition_inv_sat_trapping'], fmt="%s")
        np.savetxt("NEW_CB/sat_throats.txt", self['throat.imbibition_inv_sat_trapping'], fmt="%s")
        np.savetxt("NEW_CB/ptrap_pc.txt", ptrappc, fmt="%s")
        np.savetxt("NEW_CB/ttrap_pc.txt", ttrappc, fmt="%s")
        np.savetxt("NEW_CB/pore_volumes_trap.txt", pvol[np.where(ptrap)[0]], fmt="%s")
        np.savetxt("NEW_CB/throat_volumes_trap.txt", tvol[np.where(ptrap)[0]], fmt="%s")

        with open("NEW_CB/some_values.txt", "w") as f:
            f.write(f"Number of trapped pores: {len(np.where(ptrap)[0])}\n")
            f.write(f"Number of trapped throats: {len(np.where(ttrap)[0])}\n")
            f.write(f"trapped vol: {vol_trapped}\n")
            f.write(f"total vol: {vol_mat}\n")
            f.write(f"liquid vol: {vol_liq}\n")
            f.write(f"min(Psat trap): {min(Psat_trap)}\n")
        '''
    def evaluate_trapping_imbibition(self, p_outlets, mode='clone'):
        r"""
        Finds trapped pores and throats after a full imbibition
        percolation simulation has been run.
        Given moisture distribution, evaluate which dry pores
        are trapped/isolated (having no air path to outlet)
        at respective invasion value.
        So, it's not evaluating the to-be-wet elements!

        Parameters
        ----------
        p_outlets : array_like
            A list of pores that define the wetting phase outlets.
            Disconnection from these outlets results in trapping.

        prop_name: string
            default: 'clone' => create new properties with 'trapping' ending.
            other: 'replace' => replacing the result from run_imbibition.

        Returns
        -------
        It creates arrays called ``pore.trapped`` and ``throat.trapped``, but
        also adjusts the ``pore.prop_name`` and ``throat.prop_name`` arrays to set
        trapped locations to have zero invasion pressure.
        """
        inv_points = list(self._imbibition_pc)
        Psat_trap = sp.copy(self['pore.imbibition_inv_sat'])
        Tsat_trap = sp.copy(self['throat.imbibition_inv_sat'])
        pvol = self._net['pore.volume']
        tvol = self._net['throat.volume']
        vol_total = sp.sum(pvol) + sp.sum(tvol)
        ptrap = sp.asarray([False] * self.Np)
        ttrap = sp.asarray([False] * self.Nt)
        ptrappc = sp.zeros(self.Np)
        ttrappc = sp.zeros(self.Nt)

        pseq = self['pore.imbibition_inv_seq']
        tseq = self['throat.imbibition_inv_seq']
        Mseq = self._Mseq
        pseqtrap = Mseq*sp.ones([self.Np, ], dtype=float)
        tseqtrap = Mseq*sp.ones([self.Nt, ], dtype=float)
        net = self._net
        conns = net['throat.conns']

        for i, inv_val in enumerate(inv_points): # sp.sort()
            # Find clusters of defender pores
            Pinvaded = pseq<=i
            Tinvaded = tseq<=i
            # Cstate = conduit status: 0 = all open, 1 = 1 pore filled,
            # 2 = 2 pores filled, 3 = 2 pores + 1 throat filled
            Cstate = sp.sum(Pinvaded[conns], axis=1) + Tinvaded
            # pore clusters (Np long) determined by uninvaded conduits as active edges
            clusters = net.find_clusters(Cstate == 0)
            # Clean up clusters (Pinvaded = -1, Pdefended >=0)
            clusters = clusters * (~Pinvaded) - (Pinvaded)
            out_clusters = sp.unique(clusters[p_outlets])
            #pore clusters not connected to outlet
            trapped_pores = ~sp.in1d(clusters, out_clusters)
            trapped_pores[Pinvaded] = False
            # Maybe there are trapped throats (having 2 invaded pores)
            # except invaded throats
            Tinds = sp.logical_and(tseqtrap == Mseq, Cstate == 2)
            Tinds[sp.logical_and(Tinds, Tinvaded)] = False #Tinds -= Tinds*Tinvaded
            if sp.any(trapped_pores):
                # Pinds: dry pores becoming trapped at this inv point
                # Tinds: dry throats becoming trapped at this inv point
                # both are not to-be-wet elements! so, they dont reduce sat at
                # this step, but the previous-step trapped elements do!
                # These previous trapped elements are specified by p/tcurrent.
                # Pinds \in ~Pinvaded, pcurrent \in Pinvaded
                # sp.where(trappedpores) and still zero in pore.trapped
                Pinds = sp.logical_and(pseqtrap == Mseq, trapped_pores)
                pseqtrap[Pinds] = i
                # Trapped throats:
                trapped_throat_array = sp.asarray([False] * len(Cstate))
                # 1. all neighbor throats of a trapped pore
                trapped_throats = net.find_neighbor_throats(trapped_pores)
                trapped_throat_array[trapped_throats] = True
                Tinds = sp.logical_and(tseqtrap == Mseq, trapped_throat_array)
                # 2. all throats with 2 wet pores (Cstate==2)
                temp = sp.logical_and(tseqtrap == Mseq, Cstate == 2)
                Tinds[temp] = True
                # except invaded throats
                Tinds[sp.logical_and(Tinds, Tinvaded)] = False
                tseqtrap[Tinds] = i
                # Update list of trapped pores & throats
                ptrap[Pinds] = True
                ttrap[Tinds] = True
                ptrappc[Pinds] = inv_val
                ttrappc[Tinds] = inv_val

            elif sp.any(Tinds):
                tseqtrap[Tinds] = i
                ttrap[Tinds] = True
            # Correction to saturation: inv_sat
            pnow = sp.logical_and((pseq==i), ptrap)
            tnow = sp.logical_and((tseq==i), ttrap)
            vol_trapped = sp.sum(pvol[pnow]) + sp.sum(tvol[tnow])
            sat = vol_trapped/vol_total

            # revise other higher saturations
            pindsat = sp.logical_or(pnow, ~Pinvaded)
            tindsat = sp.logical_or(tnow, ~Tinvaded)
            Psat_trap[pindsat] = Psat_trap[pindsat] - (sat*sp.any(pindsat))
            Tsat_trap[tindsat] = Tsat_trap[tindsat] - (sat*sp.any(tindsat))

        # modify results from run_wetting with out-of-range values.
        Psat_trap[ptrap] = 10
        Tsat_trap[ttrap] = 10
        # print("vol trapped:",vol_trapped)
        print("sat:",1-sat)
        # print("t_ind:",len(np.where(ttrap)[0]))

        # self['pore.imbibition_trapped_seq'] = pseqtrap
        # self['throat.imbibition_trapped_seq'] = tseqtrap
        self['pore.imbibition_trapped_pc'] = ptrappc
        self['throat.imbibition_trapped_pc'] = ttrappc
        if mode=='replace':
            self['pore.imbibition_inv_pc'][ptrap] = 0
            self['throat.imbibition_inv_pc'][ttrap] = 0
            self['pore.imbibition_inv_seq'][ptrap] = Mseq
            self['throat.imbibition_inv_seq'][ttrap] = Mseq
            self['pore.imbibition_inv_sat'] = Psat_trap
            self['throat.imbibition_inv_sat'] = Tsat_trap
        elif mode=='clone':
            self['pore.imbibition_inv_pc_trapping'] = sp.copy(self['pore.imbibition_inv_pc'])
            self['throat.imbibition_inv_pc_trapping'] = sp.copy(self['throat.imbibition_inv_pc'])
            self['pore.imbibition_inv_pc_trapping'][ptrap] = 0.
            self['throat.imbibition_inv_pc_trapping'][ttrap] = 0.
            self['pore.imbibition_inv_seq_trapping'] = sp.copy(self['pore.imbibition_inv_seq'])
            self['throat.imbibition_inv_seq_trapping'] = sp.copy(self['throat.imbibition_inv_seq'])
            self['pore.imbibition_inv_seq_trapping'][ptrap] = Mseq
            self['throat.imbibition_inv_seq_trapping'][ttrap] = Mseq
            self['pore.imbibition_inv_sat_trapping'] = Psat_trap
            self['throat.imbibition_inv_sat_trapping'] = Tsat_trap
        else:
            raise Exception('Mode argument is either \'replace\' or \'clone\'')
        
                
        print("______OLD Aglorithm______")
        print("Number of trapped pores:", len(np.where(ptrap)[0]))
        print("Number of trapped throats:", len(np.where(ttrap)[0]))
        print("min(Psat trap):", min(Psat_trap))
        print("max(Psat trap):", max(Psat_trap))
        print("min(Tsat trap):", min(Tsat_trap))
        print("max(Tsat trap):", max(Tsat_trap))

        # print("Trapped Pores:",ptrap)
        # print("Trapped Throats:",ttrap)
        '''
        np.savetxt("OLD_CB/trapped_pores.txt", np.where(ptrap)[0], fmt="%s")
        np.savetxt("OLD_CB/trapped_throats.txt", np.where(ttrap)[0], fmt="%s")
        np.savetxt("OLD_CB/pressure_pores.txt", self['pore.imbibition_inv_pc_trapping'], fmt="%s")
        np.savetxt("OLD_CB/pressure_throats.txt", self['throat.imbibition_inv_pc_trapping'], fmt="%s")
        np.savetxt("OLD_CB/inv_seq_pores.txt", self['pore.imbibition_inv_seq_trapping'], fmt="%s")
        np.savetxt("OLD_CB/inv_seq_throat.txt", self['throat.imbibition_inv_seq_trapping'], fmt="%s")
        np.savetxt("OLD_CB/sat_pores.txt", self['pore.imbibition_inv_sat_trapping'], fmt="%s")
        np.savetxt("OLD_CB/sat_throats.txt", self['throat.imbibition_inv_sat_trapping'], fmt="%s")
        np.savetxt("OLD_CB/ptrap_pc.txt", ptrappc, fmt="%s")
        np.savetxt("OLD_CB/ttrap_pc.txt", ttrappc, fmt="%s")
        np.savetxt("OLD_CB/pore_volumes_trap.txt", pvol[np.where(ptrap)[0]], fmt="%s")
        np.savetxt("OLD_CB/throat_volumes_trap.txt", tvol[np.where(ptrap)[0]], fmt="%s")
        
        with open("OLD_CB/some_values.txt", "w") as f:
            f.write(f"Number of trapped pores: {len(np.where(ptrap)[0])}\n")
            f.write(f"Number of trapped throats: {len(np.where(ttrap)[0])}\n")
            f.write(f"trapped vol: {vol_trapped}\n")
            f.write(f"total vol: {vol_total}\n")
            f.write(f"liquid vol: {vol_total - vol_trapped}\n")
        '''


    def copy_results(self, pores=[], throats=[], phase=None,
                     prop_names=['wetting_inv_seq']):
        r"""
        Copy the results of the IP simulation into the Phase object.

        Parameters
        ----------
        pores and throats : array_like
            The list of pores and throats whose values should be returned to
            the Phase object.  Default is all of them.

        Returns
        -------
        inv_sequence : array_like
            The sequence in which each pore and throat is invaded  This depends
            on the inlet locations.  All inlets are invaded at step 0.  It is
            possible to recontruct an animation of the invasion process, in
            Paraview for instance, using this sequence information.

        """
        pores = sp.array(pores, ndmin=1)
        throats = sp.array(throats, ndmin=1)
        if len(pores) == 0:
            pores = self.Ps
        if len(throats) == 0:
            throats = self.Ts

        if phase is None:
            try:
                phase = self._phase_wet
            except:
                phase = self._phase_dry

        for prop in prop_names:
            phase['throat.'+prop] = -1.
            phase['pore.'+prop] = -1.
            phase['throat.'+prop][throats] = self['throat.'+prop][throats]
            phase['pore.'+prop][pores] = self['pore.'+prop][pores]

    def return_results(self, Pc=0, seq=None, sat=None, cases=['wetting'],
                       occupancies=['occupancy'], trapping=False,
                       phase_inv=None, phase_def=None):
        r"""
        Updates the occupancy status of invading and defending phases
        as determined by the OP algorithm

        Create the occupancy properties: pore/throat.occupancy
        """
        if phase_inv is None:
            try:
                phase_inv = self._phase_wet
            except:
                phase_inv = None

        if phase_def is None:
            try:
                phase_def = self._phase_dry
            except:
                phase_def = None

#        if 'drying' in cases[0].split('_'):
        if 'drying' in [case.split('_')[0] for case in cases]:
            phase_inv, phase_def = phase_def, phase_inv
            drying = True
        else:
            drying = False

        trap = ['' for i in cases]
        procs = cases.copy()
        if trapping:
            trap.extend(['_trapping']*len(procs))
            procs.extend(procs)
            occupancies.extend([i+'_trapping' for i in occupancies])

        for i, case in enumerate(procs):
            inv_pc = case+'_inv_pc'+trap[i]
            inv_seq = case+'_inv_seq'+trap[i]
            inv_sat = case+'_inv_sat'+trap[i]
            occupance = occupancies[i]

            if(sat is not None):
                psat = self['pore.'+inv_sat]
                tsat = self['throat.'+inv_sat]
                if drying:
                    p_inv = (psat > sat)*(psat < 10)
                    t_inv = (tsat > sat)*(tsat < 10)
                else:
                    p_inv = self['pore.'+inv_sat] <= sat
                    t_inv = self['throat.'+inv_sat] <= sat
            elif(seq is not None):
                pseq = self['pore.'+inv_seq]
                tseq = self['throat.'+inv_seq]
                # definition: sat_i = seq_i/num_seq
                if drying:
                    p_inv = (pseq > seq)*(pseq > -1)
                    t_inv = (tseq > seq)*(tseq > -1)
                else:
                    p_inv = self['pore.'+inv_seq] <= seq
                    t_inv = self['throat.'+inv_seq] <= seq
            else:
                ppc = self['pore.'+inv_pc]
                tpc = self['throat.'+inv_pc]
                if drying:
                    # all invaded pores at pc, except uninvaded pores (= wet)
                    p_inv = (ppc > Pc)*(ppc < 0)
                    t_inv = (tpc > Pc)*(tpc < 0)
                else:
                    p_inv = ppc <= Pc
                    t_inv = tpc <= Pc
            # Apply occupancy to invading phase
            phase_inv['pore.'+occupance] = p_inv*1   # change to 0 or 1
            phase_inv['throat.'+occupance] = t_inv*1
            # Apply occupancy to defending phase
            if phase_def is not None:
                phase_def['pore.'+occupance] = ~p_inv*1
                phase_def['throat.'+occupance] = ~t_inv*1

    def apply_flow(self, flowrate):
        r"""
        Convert the invaded sequence into an invaded time for a given flow rate
        considering the volume of invaded pores and throats.

        Parameters
        ----------
        flowrate : float
            The flow rate of the injected fluid

        Returns
        -------
        Creates a throat array called 'invasion_time' in the Algorithm
        dictionary

        """
        P12 = self._net['throat.conns']
        a = self['throat.inv_sequence']
        b = sp.argsort(a)
        P12_inv = self['pore.inv_sequence'][P12]
        # Find if the connected pores were invaded with or before each throat
        P1_inv = P12_inv[:, 0] == a
        P2_inv = P12_inv[:, 1] == a
        c = sp.column_stack((P1_inv, P2_inv))
        d = sp.sum(c, axis=1, dtype=bool)  # List of Pores invaded with each throat
        # Find volume of these pores
        P12_vol = sp.zeros((self.Nt,))
        P12_vol[d] = self._net['pore.volume'][P12[c]]
        # Add invaded throat volume to pore volume (if invaded)
        T_vol = P12_vol + self._net['throat.volume']
        # Cumulative sum on the sorted throats gives cumulated inject volume
        e = sp.cumsum(T_vol[b] / flowrate)
        t = sp.zeros((self.Nt,))
        t[b] = e  # Convert back to original order
        self._phase['throat.invasion_time'] = t
