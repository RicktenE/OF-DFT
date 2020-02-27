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

import math
from dolfin import *

import numpy

class TF(object):
    
    CF=(3.0/10.0)*(3.0*math.pi**2)**(2.0/3.0)
    
    def energy_density_expr(self,densobj):
        return self.CF*pow(densobj.n,(5.0/3.0))
            
    def potential_expr(self,densobj):
        return (5.0/3.0)*self.CF*pow(densobj.n,(2.0/3.0))

    def potential_enhancement_expr(self,densobj):
        return 1

    def potential_weakform(self,densobj):
        return (5.0/3.0)*self.CF*pow(densobj.n,(2.0/3.0))*densobj.v

    def energy_weakform(self,densobj):
        return self.CF*pow(densobj.n,(5.0/3.0))*densobj.v

    def energy_density_field(self,densobj, field):
        field.vector()[:] = self.CF*numpy.power(self.n.vector()[:],(5.0/3.0))
    
    def potential_field(self,densobj,field):
        field.vector()[:] = (5.0/3.0)*self.CF*numpy.power(self.n.vector()[:],(2.0/3.0))

func_tf = TF()
    