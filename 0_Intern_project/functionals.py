#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:53:14 2020

@author: rick

This code describes the functionals
"""
class TF(object):
    
    CF=(3.0/10.0)*(3.0*math.pi**2)**(2.0/3.0)
    
    def energy_density_expr(self,densobj):
        return self.CF*pow(densobj.n,(5.0/3.0))
            
    def potential_expr(self,densobj):
        return (5.0/3.0)*self.CF*pow(densobj.n,(2.0/3.0))

    def potential_enhancement_expr(self,densobj):
        return 1

    def potential_weakform(self,densobj):
        return (5.0/3.0)*self.CF*densobj.n**(2.0/3.0)*densobj.v

    def energy_weakform(self,densobj):
        return self.CF*pow(densobj.n,(5.0/3.0))*densobj.v

    def energy_density_field(self,densobj, field):
        field.vector()[:] = self.CF*np.power(self.n.vector()[:],(5.0/3.0))
    
    def potential_field(self,densobj,field):
        field.vector()[:] = (5.0/3.0)*self.CF*np.power(self.n.vector()[:],(2.0/3.0))

func_tf = TF()

class Dirac(object):
    
    CX = 3.0/4.0*(3.0/math.pi)**(1.0/3.0)
    CF = (3.0/10.0)*(3.0*math.pi**2)**(2.0/3.0)
    
    def energy_density_expr(self,densobj):
        return -self.CX*pow(densobj.n,(4.0/3.0))
            
    def potential_expr(self,densobj):
        return (-4.0/3.0)*self.CX*pow(densobj.n,1.0/3.0)

    def potential_enhancement_expr(self,densobj):
        return (-4.0/5.0)*(self.CX/self.CF)*1.0/pow(densobj.n,(1.0/3.0))

    def potential_weakform(self,densobj):
        return (-4.0/3.0)*self.CX*pow(densobj.n,(1.0/3.0))*densobj.v

    def energy_weakform(self,densobj):
        return -self.CX*pow(densobj.n,(4.0/3.0))*densobj.v

    def energy_density_field(self,densobj, field):
        field.vector()[:] = -self.CX*np.power(self.n.vector()[:],(4.0/3.0))
    
    def potential_field(self,densobj,field):
        field.vector()[:] = (-4.0/3.0)*self.CX*np.power(self.n.vector()[:],(1.0/3.0))

func_dirac = Dirac()

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
        return 1.0/8.0*dot(grad(densobj.n),grad(densobj.n))/densobj.n*densobj.v
    
    def potential_field(self,densobj,field):
        raise Exception("Not implemented.")

func_weizsacker = Weizsacker()