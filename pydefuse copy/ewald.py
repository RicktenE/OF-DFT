# -*- coding: utf-8 -*-
# 
#    This file is part of PyDeFuSE
#    Copyright (C) 2015 Rickard Armiento
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from pymatgen.analysis.ewald import EwaldSummation
from pymatgen import Lattice, Structure
from numpy import array
from math import pi

symbols = ['0', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
           'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
           'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
           'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No',
           'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Uuq', 'Uup', 'Uuh', 'Uus', 'Uuo']

pymatgen_adjust = None

def pymatgen_ewald_adjust(ewald):
    matrix = array(ewald.total_energy_matrix)

    totalcharge = 0.0
    numsites = ewald._s.num_sites
    for i in range(numsites):
        totalcharge += ewald._oxi_states[i]                

    for i in range(len(matrix)):
        matrix[i,i] = matrix[i,i] - ewald._oxi_states[i] * pi / (2.0 * ewald._vol * ewald._eta)*ewald.CONV_FACT - totalcharge**2 * pi / (2.0 * ewald._vol * ewald._eta)*ewald.CONV_FACT/numsites

    return matrix

def check_pymatgen_adjust():
    global pymatgen_adjust
    lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120,beta=90, gamma=60)
    substruct = Structure(lattice, ["H1+"],[[0,0,0]])
    test1=EwaldSummation(substruct,eta=0.20)
    totenergy1 = test1.total_energy
    test2=EwaldSummation(substruct,eta=0.15)
    totenergy2 = test2.total_energy
    #print "TEST:",totenergy1,totenergy2
    if abs(totenergy1 - totenergy2)>1e-4:
        corr_totenergy1 = pymatgen_ewald_adjust(test1)[0,0]
        corr_totenergy2 = pymatgen_ewald_adjust(test2)[0,0]
        #print "TEST:",corr_totenergy1,corr_totenergy2
        if abs(corr_totenergy1 - corr_totenergy2)<1e-4:
            pymatgen_adjust = True
        else:
            raise Exception("Pymatgen Ewald sum calculation does not behave as expected.")
    else:
        pymatgen_adjust = False

def calculate_ewald_sum(unitcell, atoms, coords,eta=None):
    global pymatgen_adjust

    unitcell = array(unitcell)*1.0/1.88971616463

    specieslist = [] 
    for atom in atoms:
        name = symbols[atom]
        specieslist += [name+str(atom)+"+"]

    lattice = Lattice(unitcell)
    struct = Structure(lattice, specieslist, coords)
    
    if pymatgen_adjust == None:
        check_pymatgen_adjust()
    ewald=EwaldSummation(struct,eta=eta)
    if pymatgen_adjust:
        matrix = pymatgen_ewald_adjust(ewald)
    else:
        matrix = ewald.total_energy_matrix
    return sum(sum(matrix))*0.036749308136659685

