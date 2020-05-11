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
from dolfin import *

class DensityFields(object):
    
    def __init__(self,n,gn2,lp):
        self.n = n
        self.gn2 = gn2
        self.lp = lp

class DensityWeakForm(object):
    
    def __init__(self,n,v):
        self.n = n
        self.v = v
        self.gradn2 = inner(grad(n),grad(n))
        self.lapln = div(grad(n))
        self.gradv = v.dx(0)

class DensityRadialWeakForm(object):
    
    def __init__(self,n,v):
        self.n = n
        self.v = v
        self.gradn = n.dx(0)
        self.gradv = v.dx(0)

    