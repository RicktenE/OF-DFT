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

class Weizsacker(object):
    
    CX = 3.0/4.0*(3.0/math.pi)**(1.0/3.0)
    CF=(3.0/10.0)*(3.0*math.pi**2)**(2.0/3.0)
    
    def energy_density_expr(self,densobj):
        return 1.0/8.0*densobj.gn2/densobj.n
            
    def potential_expr(self,densobj):
        return 1.0/8.0*densobj.gn2/pow(densobj.n,2.0) - 1.0/4.0*densobj.lp/densobj.n

    def potential_enhancement_expr(self,densobj):
        return 1.0/(8.0*self.CF)*densobj.gn2/pow(densobj.n,11.0/3.0) - 1.0/(4.0*self.CF)*densobj.lp/pow(densobj.n,8.0/3.0)

    def potential_weakform(self,densobj):
        return ((1.0/8.0*(dot(grad(densobj.n),grad(densobj.n))/(densobj.n*densobj.n))*densobj.v+(1.0/4.0*(dot(grad(densobj.n),grad(densobj.v)))/densobj.n)))

    def energy_weakform(self,densobj):
        return 1.0/8.0*inner(grad(densobj.n),grad(densobj.n))/densobj.n*densobj.v
    
    def potential_field(self,densobj,field):
        raise Exception("Not implemented.")

func_weizsacker = Weizsacker()
    