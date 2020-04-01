#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
Created on Mon Mar  2 09:27:19 2020

@author: H.R.A. ten Eikelder
program: OF-DFT radial atom solver for the TF equations.
Description: This program takes as input the TF equation and gives 
            as output the electron density of the material. In this case 
            the material is one atom. The atom will be simulated on the left 
            boundary and the 'infinite space' will be simulated on the right 
            boundary. 
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""
from __future__ import print_function
import math
import numpy as np
from dolfin import *
import matplotlib as mpl
from matplotlib import pyplot as plt


plt.close('all')

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Plot solution and mesh
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""


def plotting_solve_result(u, mesh):
    if u == u_i :
        plt.figure()
        plt.title("Last calculated Internal potential u_l")
        plt.xlabel("Radial coordinate")
        #mpl.scale.LogScale(r)
        plt.ylabel("Internal potential Vi")
        plt.grid()
        if mesh == True :
            plot(mesh)
        plot(u_i)
        
    elif u == u_n :
        plt.figure()
        plt.title("Last calculated density u_n")
        plt.xlabel("Radial coordinate")
        #mpl.scale.LogScale(r)
        plt.ylabel("Density [n]")
        plt.grid()
        if mesh == True :
            plot(mesh)
        plot(u_n)
        
    elif u == nr :
        plt.figure()
        plt.title("nr")
        plt.xlabel("Radial coordinate")
        plt.ylabel("projection of Z/(4.0*pi)*gr.dx(0)/r ")
        plt.grid()
        if mesh == True :
            plot(mesh)
        plot(nr)
    elif u == gr:
         plt.figure(1)
         plt.title("gr")
         plt.xlabel("Radial coordinate")
         plt.ylabel("projection of derivative of n towards r")
         plt.grid()
         if mesh == True :
            plot(mesh)
         plot(gr)
    else :
        print("The input to plot is invalid")
        
    # show the plots
    plt.show()
    return 
'''------------------------------------'''
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
        field.vector()[:] = self.CF*numpy.power(self.n.vector()[:],(5.0/3.0))
    
    def potential_field(self,densobj,field):
        field.vector()[:] = (5.0/3.0)*self.CF*numpy.power(self.n.vector()[:],(2.0/3.0))

func_tf = TF()

class Dirac(object):
    
    CX = 3.0/4.0*(3.0/math.pi)**(1.0/3.0)
    CF=(3.0/10.0)*(3.0*math.pi**2)**(2.0/3.0)
    
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
        field.vector()[:] = -self.CX*numpy.power(self.n.vector()[:],(4.0/3.0))
    
    def potential_field(self,densobj,field):
        field.vector()[:] = (-4.0/3.0)*self.CX*numpy.power(self.n.vector()[:],(1.0/3.0))

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

functionals = [#TF(),\
               Dirac(),\
               Weizsacker()\
               ]      
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating the mesh + Function Spaces + defining type of element
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------""" 
# Create mesh and define function space
start_x = 0
end_x = 2
amount_vertices = 100
mesh = IntervalMesh(amount_vertices,start_x, end_x) # Splits up the interval [0,1] in (n) elements 

#Creation of Function Space
V = FunctionSpace(mesh, 'P', 2) # P stands for lagrangian elemnts, number stands for degree

#Creation of mixed function space with latest update FEniCS
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
W = FunctionSpace(mesh, P1*P1)

#Define radial coordinates based on mesh
r = SpatialCoordinate(mesh)[0] # r are the x coordinates. 

#Element Kr
Z = Constant(36) # Krypton
N = Z # Neutral 


"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating and defining the boundary conditions
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""


"""def set_boundary(start_x, end_x,tol, left_value, right_value, V):
    #Defining the tolerance on the boundaries 
    tol = 1E-14

    #Defining the left boundary
    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], start_x, tol)

    #Defining the right boundary
    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], end_x, tol)

    #Defining expression on left boundary
    n_L = Expression(left_value, degree=1)         
    bc_L = DirichletBC(V, n_L, boundary_L)  
    
    #Defining expression on right boundary
    n_R = Expression(right_value, degree=1)
    bc_R = DirichletBC(V, n_R, boundary_R) 
    
    #collecting the left and right boundary in a list for the solver to read
    bcs = [bc_L, bc_R] """

#Defining the tolerance on the boundaries 
tol = 1E-14

#Defining the left boundary
def boundary_L(x, on_boundary):
    return on_boundary and near(x[0], start_x, tol)

#Defining the right boundary
def boundary_R(x, on_boundary):
    return on_boundary and near(x[0], end_x, tol)

#Defining expression on left boundary
n_L = Expression('0', degree=1)         
bc_L = DirichletBC(V, n_L, boundary_L)  
    
#Defining expression on right boundary
n_R = Expression('0', degree=1)
bc_R = DirichletBC(V, n_R, boundary_R) 
    
#collecting the left and right boundary in a list for the solver to read
bcs = [bc_L, bc_R]
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Defning external potential v[r] and Initial density n_1[r]
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
#External potential v_e[r] is analytically described for atoms 

Ex = -Z/r

#---constants---
lamb = 0.45
C1 = 3/10*(3*math.pi**2)**(2/3)
C2 = 3/4*(3/math.pi)**(1/3)
C3 = lamb/8
mu = 0

#Initial density n_i[r]
a=1/sqrt(2*pi)
b=1
#n_i = a*exp(pow((-b*(r)), 2))
n = Function(V)
n_i = Constant(0)

n = interpolate(N,V)
intn = float(assemble((n)*dx(mesh)))
print("Density integrated before adjustment:"+str(intn))
n.vector()[:] = n.vector()*N/intn  
intn = float(assemble((n)*dx(mesh)))
print("Number of electrons after adjustment:",intn)

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Defining and solving the variational problem
                    defining trial and test functions
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
print('check 1')
u_n = interpolate(n_i, V)
v_h = interpolate(Constant(-1), V)


mixed_test_functions = TestFunction(W)
(vr, pr) = split(mixed_test_functions)


print('check 2')
du_trial = TrialFunction(W)        
du = Function(W)

nlast = Function(V)
print('check 3')

u_k = Function(W)
assign(u_k.sub(0), v_h)
assign(u_k.sub(1), u_n)

bcs_du = []
print('check 4')
#--Rewriting variables for overview--
x = sqrt(r)
y= r*sqrt(u_n)
Q = r*(Ex+v_h)


eps = 1
iters = 0
maxiter = 5000
eps2 = 2

while eps > tol and iters < maxiter:
    iters += 1 
    print('check Loop top ', iters)
    #v_hk = Function(W)
    #u_nk = Function(W)
    (v_hk, u_nk) = split(u_k)
    densobj = DensityWeakForm(u_nk, pr)
    #funcpots = fucn_tf(weakdens) + func_dirac(weakdens) + func_weizsacker(weakdens)  
    funcpots = 0
    for f in functionals:
        if isinstance(f,tuple):
            funcpots += Constant(f[0]*f[1].potential_weakform(densobj))
        else:
            funcpots += f.potential_weakform(densobj)
    
    #First coupled equation: Q'' = 1/x*Q' +16pi*y^2
    F = -Q.dx(0)*vr.dx(0)*dx                                \
    + 1/x*Q.dx(0)*vr.dx(0)*dx                               \
    + 16*math.pi*y*vr*dx  
    
    # Second coupled equation y'' = ... y ... x ... Q
    F = F - y.dx(0)*pr.dx(0)*dx                             \
    + y/x*pr*dx                                             \
    + (5*C1)/(3*C3)*(y)**(7/3)/(x)**(5/3)*pr*dx             \
    - 4/3*(x)**(7/3)*(y)**(5/4)*pr*dx                       \
    + 1/C3*Q*pr*dx  
    
    #Calculate Jacobian
    J = derivative(F, u_k, du_trial)
    
    #Assemble system
    A, b = assemble_system(J,-F, bcs_du)
    
    #solve
    solve(A, du.vector(), b)
    
    print('check Loop middle', iters)
    # Conserve memory by reusing vectors for vh and n to keep dvh and dn
    dvh = vh
    vh = None
    dn = n
    n = None
    assign(dvh, du.sub(0))
    assign(dn, du.sub(1))
    
    #Calculate the Error
    avg = sum(dvh.vector().array())/len(dvh.vector().array())
    eps = numpy.linalg.norm(du.vector().array()-avg, ord=numpy.Inf)
    print('??Iteration for self-consistency:', iters,'norm:', eps)
    if math.isnan(eps):
            raise Exception("Residual error is NaN")
    
    # Conserve memory by reusing vectors for n, vh also as dn, dvh
    vh = dvh 
    dvh = None
    n = dn 
    dn = None                        
    assign(vh, u_k.sub(0))
    assign(n, u_k.sub(1))
        
    # Ad hoc negative density fix 
    omega = 1 
    nvec = n.vector()
    minval = nvec.min()
    if minval <= 0.0:
        nvec[:]=nvec[:]-minval+0.1
        intn = float(assemble((n)*dx(mesh)))
        print("Number of electrons before correction:",intn)
        nvec[:] = nvec[:]*self.N/intn    
        intn = float(assemble((field)*dx(mesh)))            
        assign(u_k.sub(1),n)
        print("Number of electrons after correction:",intn)
    
    # Calculate v_h allignment correction 
    vh_align = float(assemble((vh)*dx(mesh)))
    mu = fake_mu - vh_align
    
    n = n
    vh.vector()[:] += vh_align
    vh_align = 0.0
    vh = vh
    print('check Loop End', iters)
               
#plotting_solve_result(u_n, True)

#gr = project(u_n.dx(0),V)
#plotting_solve_result(gr, False)

#nr = project(Z/(4.0*pi)*gr.dx(0)/r,V)
#plotting_solve_result(nr, False) 


"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Saving VTKfile for post processing in ParaView
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------
#Save solution to file in VTK format
vtkfile = File('VTKfiles/radial_atom_solver_TF.pvd')
vtkfile << u_n

-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Computing the error 
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
#