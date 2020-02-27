#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys

import numpy
import pylab

try:    
    import pydefuse
except Exception:
    import inspect
    PATH_TO_PYDEFUSE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
    sys.path.insert(1, os.path.expanduser(PATH_TO_PYDEFUSE))
    import pydefuse

#coords =  [[0.0, 0.0, 0.0]]
pbc = [True, True, True]
N = None # Neutral system

# params = { 
#           'basestep':1.0,
#           'prerefine':False,
#           'prerefinepower':2.0,
#           'mesh_rectangle_divisions':4,
#           'dynrefine':False,
#           'threadpar':False,
#           'cuspcond':False,
#           'eps':1e-7,
#           'evunits':True, 
#           'gui':True,
#           'loglevel':3          
#         }

params = { 
          'basestep':1.0,
          'mesh_rectangle_divisions':12,
          'mesh_type':'jigzaw',
          'mesh_mt_refine':4,
          'mesh_mt_radial_divisions':8,
          'log_level':2
        }

functionals = [pydefuse.func_tf, pydefuse.func_dirac, pydefuse.func_weizsacker] 

print "====================================================================="
print "======== FE DFT SOLVER STARTS"
print "====================================================================="

results = []
for d in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]:
    unitcell = numpy.array([[d*1.88972612492931, 0.0, 0.0], [0.0,d*1.88972612492931,0.0], [0.0,0.0,d*1.88972612492931]])
    atoms = [1]
    coords =  [[0.5, 0.5, 0.5]]

    ofdft3d = pydefuse.Ofdft(unitcell, atoms, coords, N, pbc, functionals, **params)
    ofdft3d.run()
    ofdft3d.print_energies()
    results += [(d,ofdft3d.e_tot*27.21138386)]

print "====================================================================="
print "======== FE DFT SOLVER ENDS"
print "====================================================================="    

pydefuse.chronicle.printall()

print "Result:",results

results = numpy.array(results)
try:
    import ase
    from ase.units import kJ
    from ase.utils.eos import EquationOfState
    volumes = results[:,0]**3
    energies = results[:,1]
    eos = EquationOfState(volumes, energies)
    v0, e0, B = eos.fit()
    print(B / kJ * 1.0e24, 'GPa')
    eos.plot()
except Exception:
    raise
    pylab.xlabel(u'Volume (Å)')
    pylab.ylabel(u'Energy (eV)')
    pylab.plot(results[:,0]**3,results[:,1])
    pylab.show()
    

