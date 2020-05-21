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
import pylab


plt.close('all')

"""---------------------------------------------------------------------------"""
#Element H
#Z = 1   # Hydrogen
#N = Z 		  # Neutral

#Element Ne
#Z = Constant(10) # Neon
#N = Z 		  # Neutral

#Element Kr
Z = Constant(36) # Krypton
N = Z 		  # Neutral 

#Element 115
#Z = Constant(115)
#N = Z

a_0 = 1 # Hatree units
Alpha = (4/a_0)*(2*Z/(9*pi**2)**(1/3))


###-------Functional Constants
CF=(3.0/10.0)*(3.0*math.pi**2)**(2.0/3.0)
CX = 3.0/4.0*(3.0/math.pi)**(1.0/3.0)

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Plot function
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""

def smoothzero(x, delta):
    x[x < 10.0/delta] = 1.0/delta*np.log(1+np.exp(x[x<10.0/delta]*delta))

def smoothstep(edge0, edge1, x):
  x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0) 
  return x * x * (3 - 2 * x)

def clamp(x, lowerlimit,  upperlimit):
    return conditional(lt(x,lowerlimit),lowerlimit,conditional(gt(x,upperlimit),upperlimit,x))
    

def plotting_psi(u,title):
    a_0 = 1 # Hatree units
    Alpha_ = (4/a_0)*(2*Z/(9*pi**2)**(1/3))
    
    pylab.clf()
    rplot = mesh.coordinates()
    x = rplot*Alpha_
    y = [u(v)**(2/3)*v*(3*math.pi**2/(2*np.sqrt(2)))**(2/3)  for v in rplot]
    
    #x = numpy.logspace(-5,2,100)
    y = [u(v)**(2/3)*v*(3*math.pi**2/(2*np.sqrt(2)))**(2/3)  for v in rplot]
    pylab.plot(x,y,'kx-')
    pylab.title(title)
    pylab.pause(0.1)
    pylab.xlabel("Alpha * R")
    pylab.ylabel("Psi")

    return     

def plotting_log_keep(u,title):
    rplot = mesh.coordinates()
    x = rplot
    #x = numpy.logspace(-5,2,100)
    y = [u(v) for v in rplot]

    pylab.semilogy(x,y,'rx-')
    #pylab.plot(x,y,'kx-')
    pylab.title(title)
    pylab.pause(0.001)
#    pylab.waitforbuttonpress()
    pylab.grid
    pylab.xlabel("r")
    pylab.ylabel("n[r]")

    return 

def plotting_normal(u,title):
    pylab.clf()
    
    rplot = mesh.coordinates()
    x = rplot
    #x = numpy.logspace(-5,2,100)
    y = [u(v) for v in rplot]

    #pylab.semilogx(x,y,'bx-')
    pylab.plot(x,y,'kx-')
    pylab.title(title)
    pylab.pause(0.1)
#    pylab.waitforbuttonpress()
    pylab.grid
    pylab.xlabel("r")
    pylab.ylabel("n[r]")

    return 


def plotting_log(u,title):
    pylab.clf()
    
    rplot = mesh.coordinates()
    x = rplot
    #x = numpy.logspace(-5,2,100)
    y = [u(v) for v in rplot]

    pylab.semilogy(x,y,'bx-')
    #pylab.plot(x,y,'kx-')
    pylab.title(title)
    pylab.pause(0.001)
#    pylab.waitforbuttonpress()
    pylab.grid
    pylab.xlabel("r")
    pylab.ylabel("n[r]")

    return 


def plotting_sqrt(u,title):
    pylab.clf()
    rplot = mesh.coordinates()
    x = np.sqrt(rplot)
    #x = numpy.logspace(-5,2,100)
    y = [v*sqrt(u(v)) for v in rplot] 
    
    pylab.plot(x,y,'kx-')
    pylab.title(title)
#    pylab.waitforbuttonpress()
    pylab.pause(0.1)
    pylab.xlabel("SQRT(R)")
    pylab.ylabel("R * SQRT(density")
    
    return 
     
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating the mesh + Function Spaces + defining type of element
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------""" 


rs = np.arange(0.01,25.0,0.01)
radius = rs[-1]
r_inner = 0.0
rs_outer = [x for x in rs if x > r_inner]

start_x = rs[0]
end_x = rs[-1]
amount_vertices = len(rs)
mesh = IntervalMesh(amount_vertices,start_x, end_x) # Splits up the interval [0,1] in (n) elements 

#print("MES COORDS",np.shape(mesh.coordinates()))

#Creation of Function Space
P1 = FiniteElement('P', mesh.ufl_cell(), 2)
element = MixedElement([P1,P1])

V = FunctionSpace(mesh, 'P', 2) # P stands for lagrangian elemnts, number stands for degree
W = FunctionSpace(mesh, element)

#Define radial coordinates based on mesh
r = SpatialCoordinate(mesh)[0] # r are the x coordinates. 

"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating and defining the boundary conditions
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""

#Defining the tolerance on the boundaries 
tol = 1E-14

#Defining the left boundary
def boundary_L(x, on_boundary):
    return on_boundary and near(x[0], start_x, tol)

#Defining the right boundary
def boundary_R(x, on_boundary):
    return on_boundary and near(x[0], end_x, tol)

#Defining expression on left boundary
n_L = Expression('-10', degree=1)         
bc_L = DirichletBC(V, n_L, boundary_L)  
    
#Defining expression on right boundary
n_R = Expression('0', degree=1)
bc_R = DirichletBC(V, n_R, boundary_R) 
    
#collecting the left and right boundary in a list for the solver to read
bcs = [bc_L, bc_R]
#bcs = []
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Defning external potential v[r] 
                    Initial density n_1[r]
                    Hartree potential v_h
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
# External potential
Ex = -Z/r

startomega = 0.8
mu = 100

#########------ Initial density ------##########
#n_i = Expression('a*exp(-b*pow(x[0], 2))', degree = 2, a = 1, b=0.05)
n_i = Constant(1)

u_n = interpolate(n_i, V)

#######----- Initializing boundary conditions on Hartree potential ---###

v_hi = interpolate(Constant(-1), V)
v_h = Function(V)
u = TrialFunction(V)
v = TestFunction(V)
a = -u.dx(0)*v.dx(0)*dx 
     
L = -8.0*sqrt(2.0)/(3.0*pi)*(sqrt(Constant(mu) - Ex - v_hi))**3*v*dx\
    + (2/r)*v_hi.dx(0)*v*dx

A,  b = assemble_system(a, L, bcs)

solve(A, v_h.vector(), b)

plotting_normal(v_h, "VH  - POST first solve")

# =============================================================================
# Vvec = v_h.vector()
# for i in range(len(Vvec.get_local())):
#     if Vvec[i] > 0.0:
#         Vvec[i] = -1
# =============================================================================

plotting_normal(v_h, "VH  - POST correction")

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Defining and solving the variational problem
                    defining trial and test functions
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""

mixed_test_functions = TestFunction(W)
(qr, pr) = split(mixed_test_functions)

du_trial = TrialFunction(W)        
du = Function(W)

u_k = Function(W)
assign(u_k.sub(0), v_h)
assign(u_k.sub(1), u_n)

plotting_normal(u_k.sub(0), "VH  - POST assigning")

nlast = Function(V)


bcs_du = []


eps = 1
iters = 0
maxiter = 1000
minimal_error = 1E-10


omega = startomega
while eps > minimal_error and iters < maxiter:
    iters += 1 
    
    # Put the initial [density and v_h] in u_k
    (v_hk, u_nk) = split(u_k)
    
    plotting_normal(v_hk, "Hartree potential" )
#    plotting_log(u_nk, "In loop PRE solver")       
#    plotting_sqrt(u_nk, " Begin loop. Pre solver" )
    
    #---- Setting up functionals -------------------
    TF = (5.0/3.0)*CF*u_nk**(2.0/3.0)*pr
    DIRAC = (-4.0/3.0)*CX*pow(u_nk,(1.0/3.0))*pr
    WEIZSACKER = (1.0/8.0*(dot(grad(u_nk),grad(u_nk))/(u_nk**2)*pr+(1.0/4.0*(dot(grad(u_nk),grad(pr)))/u_nk)))
#    WEIZSACKER = (1.0/8.0*(u_nk.dx(0))*u_nk.dx(0)/(u_nk**2)*pr+(1.0/4.0*(u_nk.dx(0))*pr.dx(0))/u_nk)

    funcpots = 0
    funcpots = TF \
		+ WEIZSACKER \
        + DIRAC 
      
    # correct for possible negative mu
    correction_mu = project(Constant(mu) -Ex - u_nk,V)
    minval = correction_mu.vector().min()
    if minval < 0.0:
        mu-=minval-1e-10
        print(" New MU: ", mu)
    correction_mu = project(Constant(mu) -Ex - u_nk,V)
    print("MINIMUM VALUE",correction_mu.vector().min())
        
    
    #---- Solving v_h and u_n ----------------------
    # rotational transformation of nabla^2 v_h = -4 pi n(r)
    F = - v_hk.dx(0)*qr.dx(0)*r/2*dx    \
    + 4*math.pi*u_nk*qr*r/2*dx          \
    + v_hk.dx(0)*qr*dx             
         
    # Second coupled equation: Ts[n] + Exc[n] + Vext(r) - mu = 0
    F = F + funcpots*dx \
    + v_hk*pr*dx        \
    + Ex*pr*dx          \
    - Constant(mu)*pr*dx

    #Calculate Jacobian
    J = derivative(F, u_k, du_trial)
    
    #Assemble system
    A, b = assemble_system(J, -F, bcs_du)
    
    solve(A, du.vector(), b)
    
    
    # Conserve memory by reusing vectors for v_h and u_n to keep dv_h and du_n
    dv_h = v_h                  #step to transfer type info to dv_h
    assign(dv_h, du.sub(0))     #step to transfer values to dv_h
    v_h = None                  #empty v_h
    
    du_n = u_n                  #step to transfer type info to du_n
    assign(du_n, du.sub(1))     #step to transfer values to du_n
    u_n = None                  #empty u_n
    

    #---- Calculate the Error -----------------------
    epsexpr = conditional(lt(r,radius),du_n**2,0.0)
    eps = float(assemble((epsexpr)*dx(mesh)))    
    if math.isnan(eps):
        raise Exception("Residual error is NaN")
    print("EPS is:",eps)
    
    # Once we actually converge, we should no longer dampen the newton iterations
    if eps < 0.1:
        omega = 1.0
    
    #--- Assigning the last calculated density to nlast
    assign(nlast,u_k.sub(1))
        
    #---- Taking the step for u_n and v_h ------------
    u_k.vector()[:] = u_k.vector()[:] + omega*du.vector()[:]
        
    # Conserve memory by reusing vectors for u_n, v_h also as du_n, dv_h
    v_h = dv_h 
    assign(v_h, u_k.sub(0))
    dv_h = None
    
    u_n = du_n 
    assign(u_n, u_k.sub(1))
    du_n = None
#------------------------- electron correction    
    elecint = conditional(gt(u_n,0.0),u_n *r*r ,0.0)
    # if u_n > 0.0 elecint == u_n*r^2 
    # else          elecint == 0.0
    intn =4*math.pi*float(assemble((elecint)*ds))
    print("Electron count before correction:",intn)


    if intn <= 1e-4:
        print("Electron count too small")
        u_n = interpolate(Constant(1.0), V)

# =============================================================================
#     elif intn != N:
#         u_n.vector()[:] = u_n.vector()[:]*N/intn  
#         elecint = conditional(gt(u_n,0.0),u_n*r*r,0.0) 
#     else: 
#         u_n.vector()[:] = u_n.vector()[:]
# =============================================================================
        
    intn = 4*math.pi*float(assemble((elecint)*dx))            
    print("Number of electrons after correction:",intn)    
#--------------------------neg dens fix    
    
    plotting_log(u_n, "PRE NEG FIX")
#    plotting_sqrt(u_n, "pre neg fix")

    nvec = u_n.vector()
    minval = nvec.min()
    print("minval PRE neg fix:",minval)    
    
    x = rs_outer
    y = [u_n(rv) for rv in x]
    radius = x[-1]
    radval = 1e-8
    for i in range(len(y)):
        if y[i] <= 1e-10:
            if i == 0:
                radius = 0.0
                pass
            else:
                radius = x[i]*3.0/4.0
                radval = u_n(radius)
                break

    print("RADIUS:",radius)

    if radius == 0.0:        
        assign(u_n,interpolate(Constant(1), V))
    elif radius < x[-1]:
        fitexpr = smoothstep(radius,radius+1.0,r)*radval + (1.0-smoothstep(radius,radius+1.0,r))*conditional(gt(u_n,radval),u_n,radval)
        #conditional(gt(r,radius),1e-10,u_n)
        fitfunc = project(fitexpr, V)
        assign(u_n,fitfunc)

    #params = np.polyfit(x, np.log(y), 1)
    #fitexpr = Expression('exp(p1)*exp(p0*x[0])', degree=2, p1=params[1],p0=params[0])
    #fitexpr2 = smoothstep(rs_ob_start,rs_ob_end,r)*fitexpr + (1.0-smoothstep(rs_ob_start,rs_ob_end,r))*u_n
    #fitexpr3 = conditional(gt(fitexpr2,1e-14),fitexpr2,1e-14)
    #fitfunc = project(fitexpr2, V)
    #assign(u_n,fitfunc)

# =============================================================================
#     for i in range(len(nvec.get_local())):
#         if nvec[i] < 1e-10:
#             nvec[i] = 1e-10
# =============================================================================
            
    plotting_log_keep(u_n, "DENS POST NEG FIX")
#    plotting_sqrt(u_n, " Density post neg fix" )
    nvec = u_n.vector()
    minval = nvec.min()
    print("minval POST neg fix:",minval)    
    
    #print("u_n vector PRE neg fix: ", nvec.get_local())

    #fitexpr = conditional(gt(u_n,1e-10),u_n,1e-10)
    #fitfunc = project(fitexpr, V)
    #assign(u_n,fitfunc)

    assign(u_k.sub(1),u_n)
    print('Check end of loop, iteration: ', iters)

plotting_sqrt(nlast, " Final density") 

#----- Calculate ion-ion energy
field = Function(V)   
ionion_energy = 0.0
#----- Calculate ion-electron energy
        
field.vector()[:] = 0.0

u = TrialFunction(V)
v = TestFunction(V)
a = u.dx(0)*v.dx(0)*dx - (2/r)*u.dx(0)*v*dx
L = 4*math.pi*nlast*v*dx 
A,b = assemble_system(a, L, bcs)
# =============================================================================
#     if Z != 0:
#         print ("APPLYING DELTA",Z)
#         # TODO: Should we have delta^3? How do we do that?...
#         delta = PointSource(V, Point(fel.rs[0]),4.0*pi*Z)
#         delta.apply(b)
# =============================================================================

solve(A, field.vector(), b)
print ("= Finished")

ionelec_energy = 4*math.pi*float(assemble(field*r*r*dx))

#---- Calculate elec-elec energy
          
elecelec_energy = 0.5*float(assemble(field*r*r*dx))
        
#---- Calculate functional energy

func_energy_expression = 1.0/8.0*nlast.dx(0)/nlast\
                         + CF*pow(nlast,(5.0/3.0))\
                         - CX*pow(nlast,(4.0/3.0))

functional_energy = float(assemble(func_energy_expression*dx))
print('check type of the functional energy', type(functional_energy))
print('check type of the ionion energy', type(ionion_energy))
print('check type of the ionelec energy', type(ionelec_energy))
print('check type of the elecelec energy', type(elecelec_energy))
total = ionion_energy + ionelec_energy + elecelec_energy + functional_energy

#--- printing energies
print ("==== Resulting energies (hartree to ev): ================")
print ("Ion-ion:        % 10.4f"%(ionion_energy*27.21138386))
print ("Ion-elec:       % 10.4f"%(ionelec_energy*27.21138386))
print ("Elec-elec (Eh): % 10.4f"%(elecelec_energy*27.21138386))
print ("Functional:     % 10.4f"%(functional_energy*27.21138386))
print ("==============================================")
print ("Total energy:   % 10.4f"%(total*27.21138386))
print ("Total (Born-Oppenheimer approx):  % 10.4f"%((ionelec_energy + elecelec_energy + functional_energy)*27.21138386))
print ("==============================================")