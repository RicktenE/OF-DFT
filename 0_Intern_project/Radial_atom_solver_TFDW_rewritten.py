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
from petsc4py import *

plt.close('all')

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Plot solution and mesh
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""


def plotting_solve_result(u, mesh_bool):
    if u == v_h :
        plt.figure()
        plt.title("Last calculated Internal potential v_h")
        plt.xlabel("Radial coordinate")
        #mpl.scale.LogScale(r)
        plt.ylabel("Internal potential Vi")
        plt.grid()
        if mesh_bool == True :
            plot(mesh)
        plot(v_h)
        
    elif u == u_n :
        plt.figure()
        plt.title("Last calculated density u_n")
        plt.xlabel("Radial coordinate")
        #mpl.scale.LogScale(r)
        plt.ylabel("Density [n]")
        plt.grid()
        if mesh_bool == True :
            plot(mesh)
        plot(u_n)
        
    elif u == nr :
        plt.figure()
        plt.title("nr")
        plt.xlabel("Radial coordinate")
        plt.ylabel("projection of Z/(4.0*pi)*gr.dx(0)/r ")
        plt.grid()
        if mesh_bool == True :
            plot(mesh)
        plot(nr)
    elif u == gr:
         plt.figure(1)
         plt.title("gr")
         plt.xlabel("Radial coordinate")
         plt.ylabel("projection of derivative of n towards r")
         plt.grid()
         if mesh_bool == True :
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
        field.vector()[:] = self.CF*np.power(self.n.vector()[:],(5.0/3.0))
    
    def potential_field(self,densobj,field):
        field.vector()[:] = (5.0/3.0)*self.CF*np.power(self.n.vector()[:],(2.0/3.0))

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
end_x = 3
amount_vertices = 6
mesh = IntervalMesh(amount_vertices,start_x, end_x) # Splits up the interval [0,1] in (n) elements 

#Creation of Function Space
P1 = FiniteElement("P", mesh.ufl_cell(), 2)
element = MixedElement([P1,P1])

V = FunctionSpace(mesh, 'P', 2) # P stands for lagrangian elemnts, number stands for degree
W = FunctionSpace(mesh, element)

#Define radial coordinates based on mesh
r = SpatialCoordinate(mesh)[0] # r are the x coordinates. 

#Element Kr
#Z = Constant(36) # Krypton
#N = Z # Neutral 

#Element H
Z = Constant(1) # Hydrogen
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
Ex =Constant(1)

"""
#Fake External potential
Ex = Function(V)
rhs = Function(V)

Ex.vector()[:] =0
rhs.vector()[:]  = 0 
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v))*dx
L = rhs*v*dx
A,b = assemble_system(a, L, bcs)

solve(A, Ex.vector(), b)

print('printing Ex', Ex.vector().get_local())
"""
            

#---constants---
lamb = 0.9
C1 = 3/10*(3*math.pi**2)**(2/3)
C2 = 3/4*(3/math.pi)**(1/3)
C3 = lamb/8
mu = 0
omega = Constant(1)

#Initial density n_i[r]

n_i = Expression('a*exp(-b*pow(x[0], 2))', degree = 2, a = 1/sqrt(2*pi), b=1)
#n_i = Constant(1)
u_n = interpolate(n_i, V)
nlast = Function(V)

#------------Checking amount of electrons ----------------------
u_n = interpolate(n_i,V)
intn = float(assemble((u_n)*dx(mesh)))
print("Number of electrons before adjustment:"+str(intn))
u_n.vector()[:] = u_n.vector()*N/intn  
intn = float(assemble((u_n)*dx(mesh)))
print("Number of electrons after adjustment:",intn)

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Defining and solving the variational problem
                    defining trial and test functions
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
v_h = interpolate(Constant(-1), V)

mixed_test_functions = TestFunction(W)
(vr, pr) = split(mixed_test_functions)

du_trial = TrialFunction(W)        
du = Function(W)

u_k = Function(W)
assign(u_k.sub(0), v_h)
assign(u_k.sub(1), u_n)


bcs_du = []
eps = 1E-6
iters = 0
maxiter = 50
minimal_error = 1E-9

while eps > minimal_error and iters < maxiter:
    iters += 1 
    #######print('check begin loop', iters)
    (v_hk, u_nk) = split(u_k)
    
    #---- Rewriting variables for overview---------
    l = sqrt(r)
    y = r*sqrt(u_nk)
    Q = r*(Ex+v_hk)
    
    #---- Setting up functionals -------------------
    densobj = DensityRadialWeakForm(u_nk, pr)
    funcpots = 0
    for f in functionals:
        if isinstance(f,tuple):
            funcpots += Constant(f[0]*f[1].potential_weakform(densobj))
        else:
            funcpots += f.potential_weakform(densobj)
            
    #---- Solving v_h and u_n ----------------------
    
    #First coupled equation: Q'' = 1/l*Q' +16pi*y^2
    F = Q.dx(0)*vr.dx(0)*dx                                \
    - 1/l*Q.dx(0)*vr.dx(0)*dx                               \
    - 16*math.pi*y**2*vr*dx  
    
    # Second coupled equation y'' = ... y ... x ... Q
    F = F + y.dx(0)*pr.dx(0)*dx                             \
    - y/l*pr*dx                                             \
    - (5*C1)/(3*C3)*(y)**(7/3)/(l)**(5/3)*pr*dx             \
    + 4/3*(l)**(7/3)*(y)**(5/4)*y*pr*dx                     \
    - 1/C3*(mu*l**2-Q)*y*pr*dx  
    
    #Calculate Jacobian
    J = derivative(F, u_k, du_trial)
    
    #Assemble system
    A, b = assemble_system(J, -F, bcs_du)
       
    solve(A, du.vector(), b)
    #####print('check middle loop', iters)
    
    # Conserve memory by reusing vectors for v_h and u_n to keep dv_h and du_n
    dv_h = v_h                  #step to transfer type info to dv_h
    assign(dv_h, du.sub(0))     #step to transfer values to dv_h
    v_h = None                  #empty v_h
    
    du_n = u_n                  #step to transfer type info to du_n
    assign(du_n, du.sub(1))     #step to transfer values to du_n
    u_n = None                  #empty u_n
    
    
    #---- Calculate the Error -----------------------
    error_L2 = errornorm(du_n, dv_h, 'L2')
    eps = error_L2
    if math.isnan(eps):
            raise Exception("Residual error is NaN")

    
    #---- Taking the step for u_n and v_h ------------
    assign(nlast, u_k.sub(1))
    u_k.vector()[:] = u_k.vector()[:] + omega*du.vector()[:]
    
    # Conserve memory by reusing vectors for u_n, v_h also as du_n, dv_h
    v_h = dv_h 
    assign(v_h, u_k.sub(0))
    dv_h = None
    
    u_n = du_n 
    assign(u_n, u_k.sub(1))
    du_n = None                        
    
    ###print('check for negative density', u_n.vector().get_local())
    
    #---- Ad hoc negative density fix -------
    omega = 1 
    nvec = u_n.vector()
    minval = nvec.min()
    if minval <= 0.0:
        nvec[:]=nvec[:]-minval+1
        intn = float(assemble((u_n)*dx(mesh)))
        print("Number of electrons before correction:",intn)
        nvec[:] = nvec[:]*N/intn    
        intn = float(assemble((u_n)*dx(mesh)))            
        assign(u_k.sub(1),u_n)
        print("Number of electrons after correction:",intn)
        
    ###print('check for negative density', u_n.vector().get_local())
   
    #------- Calculate v_h allignment correction 
    vh_align = float(assemble((v_h)*dx(mesh)))
    mu_new = mu - vh_align
    
    v_h.vector()[:] += vh_align
    vh_align = 0.0
    print('Check end of loop, iteration: ', iters,' Error: ', eps)


plotting_solve_result(u_n, False)
plotting_solve_result(v_h, False)

#------------------------- calculate v_h
calc_vh = Function(V)
rhs = Function(V)
rhs.assign(u_n)
calc_vh.vector()[:]=0
    
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v))*dx
L = rhs*v*dx

A,b = assemble_system(a, L, bcs)

solve(A, calc_vh.vector(), b)
v_h = calc_vh
#---------------------------------solve for u_n from v_h
bsc = []
a = u*v*dx
L = (1.0/(4.0*pi))*inner(grad(v_h),grad(v))*dx
solve(a == L, u_n,bcs)

  
plotting_solve_result(u_n, False)
plotting_solve_result(v_h, False)















"""
Error as written in PyDeFuSe
#avg = sum(dv_h.vector().get_local())/len(dv_h.vector().get_local())
    #eps = np.linalg.norm(du.vector().get_local()-avg, ord=np.Inf)
    
     


https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/index.html 

    #---- Try to get values with numpy.array() --------
    #avg = sum(np.array(dv_h.vector())/len(np.array(dv_h.vector())))
    #eps = np.linalg.norm(np.array(du.vector())-avg, ord=np.Inf)
    #print('Iteration for self-consistency:', iters,'norm:', eps)
    #if math.isnan(eps):
    #        raise Exception("Residual error is NaN")    
            
    #---- Try to get values with gather() --------------
    #array1 = gather(dv_h)
    #array2 = gather(du)
    #avg = sum(array1())/len(array1())
    
    #----- Try to get values with VecGetArrayRead() / VecGetValues() ---- 
    #testarray = VecGetArrayRead(dv_h)
    #avg = sum(testarray)/len(testarray)
    #testarray2 = VecGetArrayRead(du_n)
    #eps = np.linalg.norm(testarray2-avg, ord=np.Inf)

    #----- As it is written in PyDeFuSe ---------------------- 
    #avg = sum(dv_h.vector().array())/len(dv_h.vector().array())
    #eps = numpy.linalg.norm(du_n.vector().array()-avg, ord=numpy.Inf)
    
    #--Negative density fix ' max step'      
    #du_nvec = du_n.vector()
    #if (du_nvec<0.0).sum() > 0:
    #    maxomega = -min(np.divide(nlast.vector()[du_nvec<0],du_n.vector()[du_nvec<0]))
    #    if maxomega > 0 and maxomega < omega:
    #        omega = maxomega/2.0



#-----------------mu_new will be used for Calculating the energies
#gr = project(u_n.dx(0),V)
#plotting_solve_result(gr, False)

#nr = project(Z/(4.0*pi)*gr.dx(0)/r,V)
#plotting_solve_result(nr, False) 

"""
"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Saving VTKfile for post processing in ParaView
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------
#Save solution to file in VTK format
vtkfile = File('VTKfiles/radial_atom_solver_TF.pvd')
vtkfile << u_n
"""