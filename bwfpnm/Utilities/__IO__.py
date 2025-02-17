import scipy as _sp
import numpy as _np
import pandas as _pd
#import yaml as _yaml
import os as _os
import itertools as _itertools
from xml.etree import ElementTree as _ET
import OpenPNM
from OpenPNM.Utilities import misc as _misc
from OpenPNM.Base import logging
from . import IO
logger = logging.getLogger(__name__)
ctrl = OpenPNM.Base.Controller()


def _to_bpnm(net, mat_size, outfile=None):
    '''Convert network object from Statoil.load to bwfpnm format'''
    macro = {'size': mat_size}

    pore = {'numbers': net.Np,
            'coords': net['pore.coords'],
            'connectivity': net['pore.connectivity'],
            'pore_neighbor': 0,
            'isInlet': net['pore.inlets'],
            'isOutlet': net['pore.outlets'],
            'throat_neighbor': 0,
            'index': 0,
            'volume': net['pore.volume'],
            'radius': net['pore.diameter']/2,
            'shapefactor': net['pore.shapefactor'],
            'clayvolume': net['pore.clayvolume']}

    throat = {'numbers': net.Nt,
              'pores': 0,
              'radius': net['throat.diameter']/2,
              'shapefactor': net['throat.shapefactor'],
              'lengthtotal': net['throat.lengthtotal'],
              'index': 0,
              'porelengths': net['throat.porelengths'],
              'length': net['throat.length'],
              'volume': net['throat.volume'],
              'clayvolume': net['throat.clayvolume'],
              'conns': net['throat.conns']}

    bc_throat = {'numbers': 0,
                 'pores': 0,
                 'radius': 0,
                 'shapefactor': 0,
                 'lengthtotal': 0,
                 'index': 0,
                 'porelengths': 0,
                 'length': 0,
                 'volume': 0,
                 'clayvolume': 0}
    IO._save_pickle(filename=outfile, macro=macro, pore=pore, throat=throat,
                 bc_throat=bc_throat)


class VTK():
    r"""
    Class for writing a Vtp file to be read by ParaView

    """

    _TEMPLATE = '''
    <?xml version="1.0" ?>
    <VTKFile byte_order="LittleEndian" type="PolyData" version="0.1">
        <PolyData>
            <Piece NumberOfLines="0" NumberOfPoints="0">
                <Points>
                </Points>
                <Lines>
                </Lines>
                <PointData>
                </PointData>
                <CellData>
                </CellData>
            </Piece>
        </PolyData>
    </VTKFile>
    '''.strip()

    @staticmethod
    def save(network, filename='', phases=[], legacy=True):
        r"""
        Save network and phase data to a single vtp file for visualizing in
        Paraview

        Parameters
        ----------
        network : OpenPNM Network Object
            The Network containing the data to be written

        filename : string, optional
            Filename to write data.  If no name is given the file is named
            after ther network

        phases : list, optional
            A list contain OpenPNM Phase object(s) containing data to be
            written

        legacy : boolean
            If True (default) the property names will be of the format
            \'pore.Cubic_asd43_diameter'\, while if False they will be
            \'pore.diameter|Cubic_asd43\'.  The latter style is consistent
            with all of the other IO methods, while the former is compatible
            with existing code, such as Paraview State files.   Eventually,
            this option will be derprecated and removed.

        """

        if filename == '':
            filename = network.name
        if ~filename.endswith('.vtp'):
            filename = filename+'.vtp'

        root = _ET.fromstring(VTK._TEMPLATE)
        objs = []
        if type(phases) != list:
            phases = [phases]
        for phase in phases:
            objs.append(phase)
        objs.append(network)
        if legacy:
            am = _misc.amalgamate_data(objs=objs)
        else:
            am = {i: network[i] for i in
                  network.props(mode=['all', 'deep']) + network.labels()}
            for phase in phases:
                dict_ = {i+'|'+phase.name: phase[i] for i in
                         phase.props(mode=['all', 'deep']) + phase.labels()}
                am.update(dict_)
        key_list = list(sorted(am.keys()))
        points = network['pore.coords']
        pairs = network['throat.conns']

        num_points = _sp.shape(points)[0]
        num_throats = _sp.shape(pairs)[0]

        piece_node = root.find('PolyData').find('Piece')
        piece_node.set("NumberOfPoints", str(num_points))
        piece_node.set("NumberOfLines", str(num_throats))

        points_node = piece_node.find('Points')
        coords = VTK._array_to_element("coords", points.T.ravel('F'), n=3)
        points_node.append(coords)

        lines_node = piece_node.find('Lines')
        connectivity = VTK._array_to_element("connectivity", pairs)
        lines_node.append(connectivity)
        offsets = VTK._array_to_element("offsets", 2*_np.arange(len(pairs))+2)
        lines_node.append(offsets)

        point_data_node = piece_node.find('PointData')
        for key in key_list:
            array = am[key]
            if array.dtype == _np.bool:
                array = array.astype(int)
            if array.size != num_points:
                continue
            element = VTK._array_to_element(key, array)
            point_data_node.append(element)

        cell_data_node = piece_node.find('CellData')
        for key in key_list:
            array = am[key]
            if array.dtype == _np.bool:
                array = array.astype(int)
            if array.size != num_throats:
                continue
            element = VTK._array_to_element(key, array)
            cell_data_node.append(element)

        tree = _ET.ElementTree(root)
        tree.write(filename)

        # Make pretty
        with open(filename, 'r+') as f:
            string = f.read()
            string = string.replace('</DataArray>', '</DataArray>\n\t\t\t')
            f.seek(0)
            # consider adding header: '<?xml version="1.0"?>\n'+
            f.write(string)

    @staticmethod
    def load(filename, network=None):
        r"""
        Read in pore and throat data from a saved VTK file.

        Parameters
        ----------
        filename : string (optional)
            The name of the file containing the data to import.  The formatting
            of this file is outlined below.

        network : OpenPNM Network Object
            The Network object onto which the data should be loaded.  If no
            Network is supplied than one will be created and returned.

        Returns
        -------
        If no Network object is supplied then one will be created and returned.
        """
        net = {}

        filename = filename.rsplit('.', maxsplit=1)[0]
        tree = _ET.parse(filename+'.vtp')
        piece_node = tree.find('PolyData').find('Piece')

        # Extract connectivity
        conn_element = piece_node.find('Lines').find('DataArray')
        array = VTK._element_to_array(conn_element, 2)
        net.update({'throat.conns': array})
        # Extract coordinates
        coord_element = piece_node.find('Points').find('DataArray')
        array = VTK._element_to_array(coord_element, 3)
        net.update({'pore.coords': array})

        # Extract pore data
        for item in piece_node.find('PointData').iter('DataArray'):
            key = item.get('Name')
            element = key.split('.')[0]
            array = VTK._element_to_array(item)
            propname = key.split('.')[1]
            net.update({element+'.'+propname: array})
        # Extract throat data
        for item in piece_node.find('CellData').iter('DataArray'):
            key = item.get('Name')
            element = key.split('.')[0]
            array = VTK._element_to_array(item)
            propname = key.split('.')[1]
            net.update({element+'.'+propname: array})

        if network is None:
            network = OpenPNM.Network.GenericNetwork()
        network = _update_network(network=network, net=net)
        return network

    @staticmethod
    def _array_to_element(name, array, n=1):
        dtype_map = {
            'int8': 'Int8',
            'int16': 'Int16',
            'int32': 'Int32',
            'int64': 'Int64',
            'uint8': 'UInt8',
            'uint16': 'UInt16',
            'uint32': 'UInt32',
            'uint64': 'UInt64',
            'float32': 'Float32',
            'float64': 'Float64',
            'str': 'String',
        }
        element = _ET.Element('DataArray')
        element.set("Name", name)
        element.set("NumberOfComponents", str(n))
        element.set("type", dtype_map[str(array.dtype)])
        element.text = '\t'.join(map(str, array.ravel()))
        return element

    @staticmethod
    def _element_to_array(element, n=1):
        string = element.text
        dtype = element.get("type")
        array = _np.fromstring(string, sep='\t')
        array = array.astype(dtype)
        if n is not 1:
            array = array.reshape(array.size//n, n)
        return array


class Statoil():
    r"""
    This class is for loading data stored in the 'Statoil' file format.  More
    specifically, this file format is used by the network extraction code of
    Blunt's group at Imperial College London, so this class can be used to load
    and work with those network.  Numerous datasets are available for download
    from the group's `website <http://tinyurl.com/zurko4q>`_.

    The so-called 'Statoil' format consists of 4 different files in a single
    folder.  The data is stored in columns with each corresponding to a
    specific property.  Headers are not provided in the files, so one must
    refer to various theses and documents to interpret their meaning.
    """
    @staticmethod
    def _no_bc(net, network=None):
        r"""
        """
        if network is None:
            network = OpenPNM.Network.GenericNetwork()
        network = _update_network(network=network, net=net)

        # Use OpenPNM Tools to clean up network
        # Trim throats connected to 'inlet' or 'outlet' reservoirs
        trim1 = _sp.where(_sp.any(net['throat.conns'] == -1, axis=1))[0]
        # Apply 'outlet' label to these pores
        pout = network['throat.conns'][trim1, 1]
        network['pore.outlets'] = False
        network['pore.outlets'][pout] = True
        trim2 = _sp.where(_sp.any(net['throat.conns'] == -2, axis=1))[0]
        # Apply 'inlet' label to these pores
        pin = network['throat.conns'][trim2, 1]
        network['pore.inlets'] = False
        network['pore.inlets'][pin] = True
        # Now trim the throats
        trim = _sp.hstack([trim1, trim2])
        network.trim(throats=trim)
        return network

    @staticmethod
    def _bc(net, size, drop_length, drop_rad, network=None, center='min',
            bccoords='original'):
        r'''
        #%% Add bcpores (artificial BOUNDARY PORES) to net['pore.*']
        # assumptions:
            # 1. The geometrical properties of the added bc pores are
            #    the same as the supplied bc throats.
            # 2. Length ratio: 1:2 (bc pore: bc throat)
            # 3. Volume is defined based on the length ratio
            # 4. The first Nbc throats are boundary throats.
                # This assumption is no longer used!
        # Configuration
        #          ----------------
        #          |               |
        #   off   on  in - BC      |
        #          |               |
        #          ----------------
        '''
        Np = net['pore.volume'].size
        conns = net['throat.conns']
        tout = _sp.where(_sp.any(conns == -1, axis=1))[0] # outlet throat
        tin = _sp.where(_sp.any(conns == -2, axis=1))[0] # inlet throat
        toutin = _sp.r_[tout, tin] # inBCpores
        if tin.size==0:
            logger.warning('The inlet pores are not specified.')
        if tout.size==0:
            logger.warning('The outlet pores are not specified.')
        # reindex the onbcpores in throat.conns
        net['throat.conns'][toutin, 0] = _sp.arange(toutin.size)+Np
        pout = conns[tout,0]    # onbc pores: outlet
        pin = conns[tin, 0]     # onbc pores: inlet
        poutin = _sp.r_[pout, pin]
        pout_in = conns[tout, 1] # outlet's inbc pores
        pin_in = conns[tin, 1] # inlet's inbc pores
        poutin_in = _sp.r_[pout_in, pin_in]

        # define the inlet-outlet direction to define the bcpore coordinates
        try:
            pos_inletin = net['pore.coords'][pin_in]
            pos_outletin = net['pore.coords'][pout_in]
            dpos = _sp.absolute(pos_outletin[:10] - pos_inletin[:10])
            dpos = _sp.average(dpos, axis=0)
            direction = _sp.where(dpos==dpos.max())[0] # flow direction: 0, 1, or 2
        except:
            direction = _sp.array([0])  # default: x direction
        adir = _sp.zeros((1, 3), dtype=bool)
        adir[0, direction] = True

        # update: pore.coords/volume/diameter/shapefactor/clayvolume of the onbc
        # pore.G/D = throat.G/D
        Npnew = conns.max() + 1
        pVclay = _sp.zeros(Npnew)
        pVclay[:Np] = net['pore.clayvolume']
        pG = _sp.zeros(Npnew)
        pG[:Np] = net['pore.shapefactor']
        pG[poutin] = net['throat.shapefactor'][toutin] # boundary tG supplied
        pD = _sp.zeros(Npnew)
        pD[:Np] = net['pore.diameter']
        pD[poutin] = net['throat.diameter'][toutin] # boundary tD supplied
#
        # define the inlet & outlet planes
        coord = net['pore.coords']
        coordmin = coord.min(axis=0)
        if center == 'mean':
            cent = _sp.mean([coord.min(axis=0), coord.max(axis=0)], axis=0)
        elif center == 'min':
            cent = (coordmin + size)/2
        inplane = cent[adir[0]] - (size[adir[0]]/2)
        outplane = cent[adir[0]] + (size[adir[0]]/2)

        # coordinates of boundary pores (forced to be in the in/outplanes)
        pcoords = _sp.zeros((Npnew, 3))
        pcoords[:Np] = net['pore.coords']
        pcoords[pout] = pos_outletin     # copy the coordinates of next-to-bc
        pcoords[pin] = pos_inletin
        tlength = net['throat.length']
        if bccoords == 'inplane':
            #%% Force the boundary pores to be in the boundary plane
            pcoords[pout, adir[0]] = outplane # modify the coords of the flow dir
            pcoords[pin, adir[0]] = inplane
        elif bccoords == 'original':
            #%% Boundary pores defined from the lengths of the boundary throats
            Lout = net['throat.porelengths'][tout, 1] + tlength[tout]
            Lin = net['throat.porelengths'][tin, 1] + tlength[tin]
            pcoords[pout, adir[0]] += Lout # modify the coords of the flow dir
            pcoords[pin, adir[0]] -= Lin

        # adjust the geometrical properties: Length (1:2)
        Lpp = _sp.linalg.norm(pcoords[poutin] - coord[poutin_in], axis=1)#L(p1tp2)

        maskr = _sp.ones_like(Lpp, dtype=bool)
        maskl = _sp.ones_like(Lpp, dtype=bool)
        if drop_rad:
            maskr = Lpp > pD[poutin_in]/2   # Lp1p2 > r_p1
        if drop_length:
            maskl = Lpp > net['throat.porelengths'][toutin, 1] # Lp1p2 > Lp1

        mask = _sp.logical_and(maskr, maskl)    # eligible bc throats

        net['throat.porelengths'][toutin[mask], 0] = tlength[toutin[mask]]/10  # L(p1)
        net['throat.length'][toutin[mask]] = tlength[toutin[mask]]*9/10        # L(t)

        tdrop= toutin[~mask]
        tindrop = tin[_sp.in1d(tin, tdrop)]
        toutdrop = tout[_sp.in1d(tout, tdrop)]
        pindrop_in = conns[tindrop, 1]
        poutdrop_in = conns[toutdrop, 1]
        pdrop = conns[tdrop, 0] # the next-to-bc pores that are at the bc plane

        # pore.volume = Lp/(Lp+Lt) * Vt; boundary Vt are supplied
        Lps = net['throat.porelengths']
        pV = _sp.zeros(Npnew)
        pV[:Np] = net['pore.volume']
        frac = Lps[toutin[mask], 0]/(Lps[toutin[mask], 0] + tlength[toutin[mask]])
        pV[poutin[mask]] = net['throat.volume'][toutin[mask]]*frac #supplied bcthroat's vol
        net['throat.volume'][toutin[mask]] -= pV[poutin[mask]]  # adjust the throat volume

        # Stich boundary pores since bc pores are represented
        # by only one index, -1 or 0. So, Np becomes Np + Nbc
        net['pore.coords'] = pcoords
        net['pore.volume'] = pV
        net['pore.diameter'] = pD
        net['pore.shapefactor'] = pG
        net['pore.clayvolume'] = pVclay
        net['pore.connectivity'] = _sp.r_[net['pore.connectivity'],
                                          _sp.ones(poutin.size)]

        # recalculate throat.lengthtotal (optional)
        coord = net['pore.coords']
        pdiff = coord[conns[:, 0]] - coord[conns[:, 1]]
        net['throat.lengthtotal'] = _sp.linalg.norm(pdiff, axis=1)

        if network is None:
            network = OpenPNM.Network.GenericNetwork()
        network = _update_network(network=network, net=net)

        pz = network.num_neighbors(pores=network.pores(), flatten=False)
        net['pore.connectivity'] = pz

        network['pore.outlets'] = False
        network['pore.outlets'][pout] = True
        network['pore.outlets'][poutdrop_in] = True

        network['pore.inlets'] = False
        network['pore.inlets'][pin] = True
        network['pore.inlets'][pindrop_in] = True

        # trim the added bcpores if their neighbor pores are at bc plane
        network.trim(pores=pdrop)
        return network

    @staticmethod
    def load(path, prefix, bcpores=True, drop_length=True, drop_rad=True,
             suffix='', network=None, outfile=None, **kwargs):
        r"""
        Load data from the \'dat\' files located in specified folder.

        Parameters
        ----------
        path : string
            The full path to the folder containing the set of \'dat\' files.

        prefix : string
            The file name prefix on each file. The data files are stored
            as \<prefix\>_node1.dat.

        network : OpenPNM Network Object
            If given then the data will be loaded on it and returned.  If not
            given, a Network will be created and return.

        """
        net = {}
        # ---------------------------------------------------------------------
        # Parse the link1 file
        def _read_link1(item='link1.dat'):
            filename = _os.path.join(path, prefix+'_'+item)
            with _read_file(filename=filename, ext='dat') as f:
                link = _pd.read_table(filepath_or_buffer=f,
                                       header=None,
                                       skiprows=1,
                                       sep=' ',
                                       skipinitialspace=True,
                                       index_col=0)
            return link

        try:
            link1 = _read_link1()
        except:
            link1 = _read_link1('Link1.DAT')
        link1.columns = ['throat.pore1', 'throat.pore2', 'throat.radius',
                         'throat.shape_factor', 'throat.total_length']
        # Add link1 props to net
        conns = _sp.vstack((link1['throat.pore1']-1,
                                          link1['throat.pore2']-1)).T
        idsort = _sp.argsort(conns, axis=1)
        isort = _sp.arange(len(conns))[:, _sp.newaxis]
        net['throat.conns'] = conns[isort, idsort]
        net['throat.diameter'] = 2 * _sp.array(link1['throat.radius'])
        net['throat.shapefactor'] = _sp.array(link1['throat.shape_factor'])
        net['throat.lengthtotal'] = _sp.array(link1['throat.total_length'])
        # ---------------------------------------------------------------------
        # Parse the link2 file
        def _read_link2(item='link2.dat'):
            filename = _os.path.join(path, prefix+'_'+item)
            with _read_file(filename=filename, ext='dat') as f:
                link = _pd.read_table(filepath_or_buffer=f,
                                       header=None,
                                       sep=' ',
                                       skipinitialspace=True,
                                       index_col=0)
            return link

        try:
            link2 = _read_link2()
        except:
            link2 = _read_link2('Link2.DAT')
        link2.columns = ['throat.pore1', 'throat.pore2',
                         'throat.pore1_length', 'throat.pore2_length',
                         'throat.length', 'throat.volume',
                         'throat.clayvolume']
        # Add link2 props to net
        net['throat.length'] = _sp.array(link2['throat.length'])
        net['throat.volume'] = _sp.array(link2['throat.volume'])
        net['throat.clayvolume'] = _sp.array(link2['throat.clayvolume'],
                                              dtype=float)
        porelengths = _sp.vstack((link2['throat.pore1_length'],
                                 link2['throat.pore2_length'])).T
        net['throat.porelengths'] = porelengths[isort, idsort]
        # ---------------------------------------------------------------------
        # Parse the node1 file
        def _read_node1(item='node1.dat'):
            filename = _os.path.join(path, prefix+'_'+item)
            with _read_file(filename=filename, ext='dat') as f:
                row_0 = f.readline().split()
                num_lines = int(row_0[0])
                size = _sp.array([float(i) for i in row_0[1:]])
                array = _sp.ndarray([num_lines, 6])
                for i in range(num_lines):
                    row = f.readline()\
                           .replace('\t', ' ').replace('\n', ' ').split()
                    array[i, :] = row[0:6]
            return array, size

        try:
            array, size = _read_node1()
        except:
            array, size = _read_node1('Node1.DAT')
        node1 = _pd.DataFrame(array[:, [1, 2, 3, 4]])
        node1.columns = ['pore.x_coord', 'pore.y_coord', 'pore.z_coord',
                         'pore.coordination_number']
        # Add node1 props to net
#        net['prop.Lmat'] = size
        net['pore.connectivity'] = node1['pore.coordination_number']
        net['pore.coords'] = _sp.vstack((node1['pore.x_coord'],
                                         node1['pore.y_coord'],
                                         node1['pore.z_coord'])).T
        # ---------------------------------------------------------------------
        # Parse the node2 file
        def _read_node2(item='node2.dat'):
            filename = _os.path.join(path, prefix+'_'+item)
            with _read_file(filename=filename, ext='dat') as f:
                node = _pd.read_table(filepath_or_buffer=f,
                                       header=None,
                                       sep=' ',
                                       skipinitialspace=True,
                                       index_col=0)
            return node

        try:
            node2 = _read_node2()
        except:
            node2 = _read_node2('Node2.DAT')

        node2.columns = ['pore.volume', 'pore.radius', 'pore.shapefactor',
                         'pore.clayvolume']
        # Add node2 props to net
        net['pore.volume'] = _sp.array(node2['pore.volume'])
        net['pore.diameter'] = 2 * _sp.array(node2['pore.radius'])
        net['pore.shapefactor'] = _sp.array(node2['pore.shapefactor'])
        net['pore.clayvolume'] = _sp.array(node2['pore.clayvolume'],
                                            dtype=float)
        if not bcpores:
            network = Statoil._no_bc(net)
        else:
            network = Statoil._bc(net, size, drop_length, drop_rad, **kwargs)
        # ---------------------------------------------------------------------
        if suffix != '':
            if outfile is None:
                outfile = _os.path.join(path, prefix+'_'+suffix+'.p')
        else:
            if outfile is None:
                outfile = _os.path.join(path, prefix+'.p')
        _to_bpnm(network, mat_size=size, outfile=outfile)

        return network




class Jiang():
    r"""
    This class is for loading data stored in the 'Jiang' file format, slightly
    different from Statoil format.

    ================= Jiang's data format ===========================
    Link1: (7) index, pore 1 index, pore 2 index, inscribed radius, (hydraulic radius), shape factor, total length (pore centre to pore centre) pore index -1 inlet, 0 outlet

    Link2:    index, pore 1 index, pore 2 index, length pore 1, length pore 2, length throat, volume, clay volume  length pore: half length, i.e. from pore centre to start of throat

    Node1:  index, x_pos, y_pos, z_pos, connection num, connecting nodes..., at inlet?, at outlet?, connecting links...

    Node2:  index, volume, radius, (hydraulic radius), shape factor, clay volume
    """

    @staticmethod
    def load(path, prefix, suffix='', network=None):
        r"""
        Load data from the \'dat\' files located in specified folder.

        Parameters
        ----------
        path : string
            The full path to the folder containing the set of \'DAT\' files.

        prefix : string
            The file name prefix on each file. The data files are stored
            as \<prefix\>_Node1.DAT.

        network : OpenPNM Network Object
            If given then the data will be loaded on it and returned.  If not
            given, a Network will be created and return.

        """
        net = {}

        # ---------------------------------------------------------------------
        # Parse the link1 file
        for item in ['Link1']:
            filename = _os.path.join(path, prefix+'_'+item+'.DAT')
            with _read_file(filename=filename, ext='DAT') as f:
                link1 = _pd.read_table(filepath_or_buffer=f,
                                       header=None,
                                       skiprows=1,
                                       sep=' ',
                                       skipinitialspace=True,
                                       index_col=0)
        link1.columns = ['throat.pore1', 'throat.pore2', 'throat.radius',
                         'throat.radius_hydraulic',
                         'throat.shapefactor', 'throat.total_length']
        # Add link1 props to net
        conns = _sp.vstack((link1['throat.pore1']-1,
                                          link1['throat.pore2']-1)).T
        idsort = _sp.argsort(conns, axis=1)
        isort = _sp.arange(len(conns))[:, _sp.newaxis]
        net['throat.conns'] = conns[isort, idsort]
        net['throat.diameter'] = 2 * _sp.array(link1['throat.radius'])
        net['throat.shapefactor'] = _sp.array(link1['throat.shapefactor'])
        net['throat.lengthtotal'] = _sp.array(link1['throat.total_length'])
        # ---------------------------------------------------------------------
        # Parse the link2 file
        for item in ['Link2']:
            filename = _os.path.join(path, prefix+'_'+item+'.DAT')
            with _read_file(filename=filename, ext='DAT') as f:
                link2 = _pd.read_table(filepath_or_buffer=f,
                                       header=None,
                                       sep=' ',
                                       skipinitialspace=True,
                                       index_col=0)
        link2.columns = ['throat.pore1', 'throat.pore2',
                         'throat.pore1_length', 'throat.pore2_length',
                         'throat.length', 'throat.volume',
                         'throat.clayvolume']
        # Add link2 props to net
        net['throat.length'] = _sp.array(link2['throat.length'])
        net['throat.volume'] = _sp.array(link2['throat.volume'])
        net['throat.clayvolume'] = _sp.array(link2['throat.clayvolume'],
                                              dtype=float)
        porelengths = _sp.vstack((link2['throat.pore1_length'],
                                 link2['throat.pore2_length'])).T
        net['throat.porelengths'] = porelengths[isort, idsort]
        # ---------------------------------------------------------------------
        # Parse the node1 file
        for item in ['Node1']:
            filename = _os.path.join(path, prefix+'_'+item+'.DAT')
            with _read_file(filename=filename, ext='DAT') as f:
                row_0 = f.readline().split(' ')
                while '' in row_0:
                    row_0.remove('')
                num_lines = int(row_0[0])
                size = _sp.array([float(i) for i in row_0[1:]])
                array = _sp.ndarray([num_lines, 6])
                for i in range(num_lines):
                    row = f.readline().split(' ')
                    while '' in row:
                        row.remove('')
                    try:
                        row.remove('\n')
                    except:
                        pass
                    array[i, :] = row[0:6]
        node1 = _pd.DataFrame(array[:, [1, 2, 3, 4]])
        node1.columns = ['pore.x_coord', 'pore.y_coord', 'pore.z_coord',
                         'pore.coordination_number']
        # Add node1 props to net
        net['pore.connectivity'] = node1['pore.coordination_number']
        net['pore.coords'] = _sp.vstack((node1['pore.x_coord'],
                                         node1['pore.y_coord'],
                                         node1['pore.z_coord'])).T
        # ---------------------------------------------------------------------
        # Parse the node2 file
        for item in ['Node2']:
            filename = _os.path.join(path, prefix+'_'+item+'.DAT')
            with _read_file(filename=filename, ext='DAT') as f:
                node2 = _pd.read_table(filepath_or_buffer=f,
                                       header=None,
                                       sep=' ',
                                       skipinitialspace=True,
                                       index_col=0)
        node2.columns = ['pore.volume', 'pore.radius',
                         'throat.radius_hydraulic', 'pore.shapefactor',
                         'pore.clayvolume']
        # Add node2 props to net
        net['pore.volume'] = _sp.array(node2['pore.volume'])
        net['pore.diameter'] = 2 * _sp.array(node2['pore.radius'])
        net['pore.shapefactor'] = _sp.array(node2['pore.shapefactor'])
        net['pore.clayvolume'] = _sp.array(node2['pore.clayvolume'],
                                            dtype=float)
        # ---------------------------------------------------------------------
        if network is None:
            network = OpenPNM.Network.GenericNetwork()
        network = _update_network(network=network, net=net)

        # Use OpenPNM Tools to clean up network
        # Trim throats connected to 'inlet' or 'outlet' reservoirs
        trim1 = _sp.where(_sp.any(net['throat.conns'] == -1, axis=1))[0]
        # Apply 'outlet' label to these pores
        outlets = network['throat.conns'][trim1, 1]
        network['pore.outlets'] = False
        network['pore.outlets'][outlets] = True
        trim2 = _sp.where(_sp.any(net['throat.conns'] == -2, axis=1))[0]
        # Apply 'inlet' label to these pores
        inlets = network['throat.conns'][trim2, 1]
        network['pore.inlets'] = False
        network['pore.inlets'][inlets] = True
        # Now trim the throats
        trim = _sp.hstack([trim1, trim2])
        network.trim(throats=trim)

        if suffix != '':
            outfile = _os.path.join(path, prefix+'_'+suffix+'.p')
        else:
            outfile = _os.path.join(path, prefix+'.p')
        _to_bpnm(network, mat_size=size,
                 outfile=outfile)

        return network


class MAT():
    r"""
    Class for reading and writing OpenPNM data to a Matlab 'mat' file

    Notes
    -----
    The 'mat' file must contain data formatted as follows:

    1. The file can contain either or both pore and throat data.

    2. The property names should be in the format of ``pore_volume`` or
    ``throat_surface_area`. In OpenPNM the first \'_\' will be replaced by
    a \'.\' to give \'pore.volume\' or \'throat.surface_area\'.

    3. Boolean data represented as 1's and 0's will be converted to the
    Python boolean True and False.  These will become \'labels\' in
    OpenPNM.
    """

    @staticmethod
    def save(network, filename='', phases=[]):
        r"""
        Write Network to a Mat file for exporting to Matlab.

        Parameters
        ----------
        network : OpenPNM Network Object

        filename : string
            Desired file name, defaults to network name if not given

        phases : list of phase objects ([])
            Phases that have properties we want to write to file

        """
        if filename == '':
            filename = network.name
        filename = filename.replace('.mat', '') + '.mat'
        if type(phases) is not list:  # Ensure it's a list
            phases = [phases]

        keys = network.props(mode=['all', 'deep']) + network.labels()
        pnMatlab = {i.replace('.', '_'): network[i] for i in keys}

        for phase in phases:
            keys = phase.props(mode=['all', 'deep']) + phase.labels()
            temp = {i.replace('.', '_')+'|'+phase.name: phase[i]
                    for i in keys}
            pnMatlab.update(temp)

        _sp.io.savemat(file_name=filename, mdict=pnMatlab)

    @staticmethod
    def load(filename, network=None):
        r"""
        Loads data onto the given network from an appropriately formatted
        'mat' file (i.e. MatLAB output).

        Parameters
        ----------
        filename : string (optional)
            The name of the file containing the data to import.  The formatting
            of this file is outlined below.

        network : OpenPNM Network Object
            The Network object onto which the data should be loaded.  If no
            Network is supplied than one will be created and returned.

        Returns
        -------
        If no Network object is supplied then one will be created and returned.

        """
        net = {}

        import scipy.io as _spio
        data = _spio.loadmat(filename)
        # Deal with pore coords and throat conns specially
        if 'throat_conns' in data.keys():
            net.update({'throat.conns': _sp.vstack(data['throat_conns'])})
            Nt = _sp.shape(net['throat.conns'])[0]
            net.update({'throat.all': _sp.ones((Nt,), dtype=bool)})
            del data['throat_conns']
        else:
            logger.warning('\'throat_conns\' not found')
        if 'pore_coords' in data.keys():
            net.update({'pore.coords': _sp.vstack(data['pore_coords'])})
            Np = _sp.shape(net['pore.coords'])[0]
            net.update({'pore.all': _sp.ones((Np,), dtype=bool)})
            del data['pore_coords']
        else:
            logger.warning('\'pore_coords\' not found')

        # Now parse through all the other items
        items = [i for i in data.keys() if '__' not in i]
        for item in items:
            element = item.split('_')[0]
            prop = item.split('_', maxsplit=1)[1]
            net[element+'.'+prop] = _sp.squeeze(data[item].T)

        if network is None:
            network = OpenPNM.Network.GenericNetwork()
        network = _update_network(network=network, net=net)
        return network


class Pandas():

    @staticmethod
    def get_data_frames(network, phases=[]):
        r"""
        Convert the Network (and optionally Phase) data to Pandas DataFrames.

        Parameters
        ----------
        network : OpenPNM Network Object
            The Network containing the data to be stored

        phases : list of OpenPNM Phase Objects
            The data on each supplied phase will be added to the CSV file

        Returns
        -------
        A dict containing 2 Pandas DataFrames with 'pore' and 'throat' data in
        each.
        """
        if type(phases) is not list:  # Ensure it's a list
            phases = [phases]

        # Initialize pore and throat data dictionary with conns and coords
        pdata = {}
        tdata = {}

        # Gather list of prop names from network and geometries
        pprops = set(network.props(element='pore', mode=['all', 'deep']) +
                     network.labels(element='pore'))
        tprops = set(network.props(element='throat', mode=['all', 'deep']) +
                     network.labels(element='throat'))

        # Select data from network and geometries using keys
        for item in pprops:
            pdata.update({item: network[item]})
        for item in tprops:
            tdata.update({item: network[item]})

        # Gather list of prop names from phases and physics
        for phase in phases:
            # Gather list of prop names
            pprops = set(phase.props(element='pore', mode=['all', 'deep']) +
                         phase.labels(element='pore'))
            tprops = set(phase.props(element='throat', mode=['all', 'deep']) +
                         phase.labels(element='throat'))
            # Add props to tdata and pdata
            for item in pprops:
                pdata.update({item+'|'+phase.name: phase[item]})
            for item in tprops:
                tdata.update({item+'|'+phase.name: phase[item]})

        # Scan data and convert non-1d arrays to strings
        for item in list(pdata.keys()):
            if _sp.shape(pdata[item]) != (network.Np,):
                array = pdata.pop(item)
                temp = _sp.empty((_sp.shape(array)[0], ), dtype=object)
                for row in range(temp.shape[0]):
                    temp[row] = str(array[row, :]).strip('[]')
                pdata.update({item: temp})

        for item in list(tdata.keys()):
            if _sp.shape(tdata[item]) != (network.Nt,):
                array = tdata.pop(item)
                temp = _sp.empty((_sp.shape(array)[0], ), dtype=object)
                for row in range(temp.shape[0]):
                    temp[row] = str(array[row, :]).strip('[]')
                tdata.update({item: temp})

        data = {'pore.DataFrame': _pd.DataFrame.from_dict(pdata),
                'throat.DataFrame': _pd.DataFrame.from_dict(tdata)}

        return data


class CSV():
    r"""
    This class is used for reading and writing CSV files containing pore and
    throat property data.  This class uses Pandas for transferring data from
    the OpenPNM format to CSV.

    Notes
    -----
    There are a few rules governing how the data should be stored:

    1. The first row of the file (column headers) must contain the
    property names. The subsequent rows contain the data.

    2. The property names should be in the usual OpenPNM format, such as
    of *pore.volume* or *throat.surface_area*.

    3. Each column represents a specific property.  For Np x 1 or Nt x 1
    data such as *pore.volume* this is straightforward.  For Np x m or
    Nt x m data, it must be entered in as a set of values NOT separated by
    commas.  For instance, the *pore.coords* values should be X Y Z with
    spaces, not commas between them.

    4. The file can contain both or either pore and throat data.

    5. Labels can be imported by placing the characters TRUE and FALSE
    in a column corresponding to the label name (i.e. *pore.front*).  TRUE
    indicates where the label applies and FALSE otherwise.
    """

    @staticmethod
    def save(network, filename='', phases=[]):
        r"""
        Save all the pore and throat property data on the Network (and
        optionally on any Phases objects) to CSV files.

        Parameters
        ----------
        network : OpenPNM Network Object
            The Network containing the data to be stored

        filename : string
            The name of the file to store the data

        phases : list of OpenPNM Phase Objects
            The data on each supplied phase will be added to the CSV file.

        Notes
        -----
        The data from all Geometry objects is added to the file automatically.
        Furthermore, the Physics data is added for each Phase object that is
        provided.
        """
        if type(phases) is not list:  # Ensure it's a list
            phases = [phases]

        dataframes = Pandas.get_data_frames(network=network, phases=phases)
        dfp = dataframes['pore.DataFrame']
        dft = dataframes['throat.DataFrame']
        b = dft.join(other=dfp, how='left')

        # Write to file
        if filename == '':
            filename = network.name
        with _write_file(filename=filename, ext='csv') as f:
            b.to_csv(f, index=False)

    @staticmethod
    def load(filename, network=None):
        r"""
        Opens a 'csv' file, reads in the data, and adds it to the **Network**

        Parameters
        ----------
        filename : string (optional)
            The name of the file containing the data to import.  The formatting
            of this file is outlined below.

        Returns
        -------
        If no Network object is supplied then one will be created and returned.

        """
        net = {}

        with _read_file(filename=filename, ext='csv') as f:
            a = _pd.read_table(filepath_or_buffer=f,
                               sep=',',
                               skipinitialspace=True,
                               index_col=False,
                               true_values=['T', 't', 'True', 'true',
                                            'TRUE'],
                               false_values=['F', 'f', 'False', 'false',
                                             'FALSE'])

        # Now parse through all the other items
        for item in a.keys():
            element = item.split('.')[0]
            prop = item.split('.', maxsplit=1)[1]
            data = _sp.array(a[item].dropna())
            if type(data[0]) is str:
                N = _sp.shape(data)[0]
                if '.' in data[0].split(' ')[0]:  # Decimal means float
                    dtype = float
                else:
                    dtype = int
                temp = _sp.empty(_sp.shape(data), dtype=object)
                for row in range(N):
                    temp[row] = _sp.fromstring(data[row], sep=' ', dtype=dtype)
                data = _sp.vstack(temp)
            else:
                dtype = type(data[0])
            net[element+'.'+prop] = data.astype(dtype)

        if network is None:
            network = OpenPNM.Network.GenericNetwork()
        network = _update_network(network=network, net=net)
        return network


def _update_network(network, net):
    # Infer Np and Nt from length of given prop arrays in file
    for element in ['pore', 'throat']:
        N = [_sp.shape(net[i])[0] for i in net.keys() if i.startswith(element)]
        if N:
            N = _sp.array(N)
            if _sp.all(N == N[0]):
                if (network._count(element) == N[0]) \
                        or (network._count(element) == 0):
                    network.update({element+'.all': _sp.ones((N[0],),
                                                             dtype=bool)})
                    net.pop(element+'.all', None)
                else:
                    raise Exception('Length of '+element+' data in file ' +
                                    'does not match network')
            else:
                raise Exception(element+' data in file have inconsistent ' +
                                'lengths')

    # Add data on dummy net to actual network
    for item in net.keys():
        # Try to infer array types and change if necessary
        # Chcek for booleans disguised and 1's and 0's
        num0s = _sp.sum(net[item] == 0)
        num1s = _sp.sum(net[item] == 1)
        if (item.split('.')[-1] not in ['clayvolume', 'conns']):
            if ((num1s + num0s) == _sp.shape(net[item])[0]):
                net[item] = net[item].astype(bool)
        # Write data to network object
        if item not in network:
            network.update({item: net[item]})
        else:
            logger.warning('\''+item+'\' already present')
    return network


def _write_file(filename, ext):
    ext = ext.replace('.', '').lower()
    if ext not in ['csv', 'yaml', 'mat', 'vtp', 'dat', 'DAT']:
        raise Exception(ext+' is not a supported file extension')
    filename = filename.rstrip('.'+ext)
    filename = filename+'.'+ext
    try:
        logger.warning(filename+' already exists, contents will be ' +
                       'overwritten')
        f = open(filename, mode='w')
    except:
        f = open(filename, mode='x')
    return f


def _read_file(filename, ext):
    ext = ext.replace('.', '').lower()
    if ext not in ['csv', 'yaml', 'mat', 'vtp', 'dat', 'DAT']:
        raise Exception(ext+' is not a supported file extension')
    if not filename.lower().endswith('.'+ext):
        filename = filename+'.'+ext
    f = open(filename, mode='r')
    return f
