#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math, time, os, sys
import numpy, pylab

try:    
    import pydefuse
except Exception:
    import inspect
    PATH_TO_PYDEFUSE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
    sys.path.insert(1, os.path.expanduser(PATH_TO_PYDEFUSE))
    import pydefuse

a=2.70230835864891
unitcell = numpy.array([[a, 0.0, 0.0], [0.0,a,0.0], [0.0,0.0,a]])
atoms = [4]
# Note: all reduced coordinates must be [0,1)
coords =  [[0.5, 0.5, 0.5]]
pbc = [True, True, True]
N = None

functionals = [pydefuse.func_tf, pydefuse.func_dirac, pydefuse.func_weizsacker] 

allresults={}

results = []
for refinement in range(10):

#     params = { 
#               'element_order':1,
#               'basestep':1.0,
#               'mesh_rectangle_divisions':2+2*refinement,
#               'mesh_type':'jigzaw',
#               'mesh_mt_refine':4,
#               'mesh_mt_radial_divisions':refinement,
#               'dynrefine':False,
#               'threadpar':False,
#               'cuspcond':False,
#               'eps':5e-5,
#               'evunits':True, 
#               'gui':False,
#               'loglevel':3,     
#             }

#     params = { 
#                'element_order':2,
#                'basestep':1.0,
#                'prerefine':False,
#                'prerefinepower':2.0,
#                'mesh_rectangle_divisions':4+2*refinement,
#                'dynrefine':False,
#                'threadpar':False,
#                'cuspcond':False,
#                'eps':1e-7,
#                'evunits':True, 
#                'gui':False,
#               }

    params = { 
               'element_order':1,
               'basestep':1.0,
               'prerefine':False,
               'prerefinepower':2.0,
               'mesh_rectangle_divisions':4+2*refinement,
               'dynrefine':False,
               'threadpar':False,
               'cuspcond':False,
               'eps':1e-7,
               'evunits':True, 
               'gui':False,
              }
                      
    ofdft3d = pydefuse.Ofdft(unitcell, atoms, coords, N, pbc, functionals, **params)
    start = time.clock()
    ofdft3d.run()
    stop = time.clock()

    ofdft3d.print_energies()
    results += [(refinement,ofdft3d.e_tot*27.21138386,ofdft3d.fel.msh.num_cells(),stop-start)]

    allresults['results'] = results

    pylab.figure(2)
    pylab.clf()
    pylab.xlabel(u'1/cells')
    pylab.ylabel(u'Energy (eV)')
    for name in allresults:    
        data = numpy.array(allresults[name])
        pylab.plot(1.0/(0.001*data[:,2]),data[:,1],label=name)
    pylab.legend()

    pylab.figure(3)
    pylab.clf()
    pylab.xlabel(u'Time (s)')
    pylab.ylabel(u'Energy (eV)')
    for name in allresults:    
        data = numpy.array(allresults[name])
        pylab.plot(data[:,3],data[:,1],'x-',label=name)
    pylab.legend()
    pylab.pause(0.0001)
    
pydefuse.chronicle.printall()

print "Result:",results

pylab.show()
