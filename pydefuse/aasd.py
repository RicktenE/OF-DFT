#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:11:18 2020

@author: rick
"""


import os, sys

import numpy

    
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

functionals = [func_tf, func_dirac, (1.0/9.0, func_weizsacker)] 

print ("* FE OF-DFT SOLVER STARTS")

ofdft3d = Ofdft(unitcell, atoms, coords, N, pbc, functionals, **params)
ofdft3d.run()

print ("* FE OF-DFT SOLVER ENDS")

ofdft3d.print_energies()
chronicle.printall()

ofdft3d.show_density()
ofdft3d.show_vh()