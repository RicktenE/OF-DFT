#!/usr/bin/env python
import os, sys

import numpy

try:    
    import pydefuse
except Exception:
    import inspect
    PATH_TO_PYDEFUSE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
    sys.path.insert(1, os.path.expanduser(PATH_TO_PYDEFUSE))
    import pydefuse
    
a=2.70230835864891
unitcell = numpy.array([[a, 0.0, 0.0], [0.0,a,0.0], [0.0,0.0,a]])
atoms = [36]
# Note: all reduced coordinates must be [0,1)
coords =  [[0.5, 0.5, 0.5]]
#coords =  [[0.0, 0.0, 0.0]]
pbc = [True, True, True]
N = None # Neutral system

params = { 
          'basestep':1.0,
          'mesh_rectangle_divisions':6,
          'mesh_type':'jigzaw',
          'mesh_mt_refine':3,
          'mesh_mt_radial_divisions':3,
          'log_level':2
        }

functionals = [pydefuse.func_tf, pydefuse.func_dirac, (1.0/9.0, pydefuse.func_weizsacker)] 

print "* FE OF-DFT SOLVER STARTS"

ofdft3d = pydefuse.Ofdft(unitcell, atoms, coords, N, pbc, functionals, **params)
ofdft3d.run()

print "* FE OF-DFT SOLVER ENDS"

ofdft3d.print_energies()
pydefuse.chronicle.printall()

ofdft3d.show_density()
ofdft3d.show_vh()

