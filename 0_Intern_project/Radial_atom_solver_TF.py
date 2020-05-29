#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
Created on Mon Mar  2 09:27:19 2020

@author: H.R.A. ten Eikelder
program: OF-DFT radial atom solver using the FEniCS Python module. 
Description: This program takes as input the DFT equations and gives 
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
from fenics import *
from matplotlib import pyplot as plt
import pylab

plt.close('all')

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Plot function
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
  

def plotting_psi(u,title):
    pylab.clf()
    
    rplot = mesh.coordinates()
    x = Alpha* rplot
    #x = numpy.logspace(-5,2,100)
    y = [u(v) for v in rplot]

    pylab.plot(x,y,'bx-')
    pylab.title(title)
    pylab.pause(1)
    pylab.grid
    pylab.xlabel("Alpha * R")
    pylab.ylabel("Psi")

    return 
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating the mesh 
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------""" 
# Create mesh and define function space
start_x = 0.0
end_x = 25
amount_vertices = 200
mesh = IntervalMesh(amount_vertices,start_x, end_x) # Splits up the interval [x_start,x_end] in (n) elements 
V = FunctionSpace(mesh, 'P', 1) # P stands for lagrangian elemnts, number stands for degree
r = SpatialCoordinate(mesh)[0] # r are the x coordinates. 

'''----------------------constants-------------------------------'''

a_0 = 1 # if all natural constants are 1
Z = 1 # if all natural constants are 1
Alpha = 4/a_0*(2*Z/(9*pi**2))**(1/3)
mu = 0

"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Defning external potential v[r] and Initial density n_1[r]
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
#External potential v_e[r] is analytically described for atoms 
Ex = -Z/r

#Initial density n_1[r]
#n_i = Expression('a*exp(pow((-b*(x[0])), 2))', degree =1, a =1/sqrt(2*pi), b=1)
n_i = Constant(0)

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

#defining expression on left boundary
n_L = Expression('1', degree=1)         #Eq. 5,189
bc_L = DirichletBC(V, n_L, boundary_L)  

#defining expression on right boundary
n_R = Expression('0', degree=1)         # Eq. 5,190
bc_R = DirichletBC(V, n_R, boundary_R) 

#collecting the left and right boundary in a list for the solver to read
bcs = [bc_L, bc_R] 


"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Defining and solving the variational problem
                    defining trial and test functions
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
u_i = interpolate(n_i, V)  # previous (known) u
u = TrialFunction(V)
v = TestFunction(V)
a = -u.dx(0)*v.dx(0)*dx 
L = Alpha**(-1/2.0)*(1.0/(sqrt(r)))*u_i**(3/2)*v*dx
A,  b = assemble_system(a, L, bcs)
u_k = Function(V)
solve(A, u_k.vector(), b)

#redefine boundary conditions for loop iteration
bc_L_du = DirichletBC(V, Constant(0), boundary_L)
bc_R_du = DirichletBC(V, Constant(0), boundary_R)
bcs_du = [bc_L_du, bc_R_du]


du_ = TrialFunction(V)

# Start second solve
du = Function(V)
u  = Function(V)  # u = u_k + omega*du
omega = 1.0       # relaxation parameter
eps = 1.0


#Initiate loop for convergence on value for u
iter = 0
maxiter = 500
while eps > 1e-11 and iter < maxiter:
    iter += 1
    
    #Redifine trial function and derivative of function
    F  = -u_k.dx(0)*v.dx(0)*dx - Alpha**(-1/2.0)*(1.0/(sqrt(r)))*u_k**(3/2)*v*dx
    J = derivative(F, u_k, du_)
    plotting_psi(u_k,'Density - PSI')

    A, b = assemble_system(J, -F, bcs_du)
    solve(A, du.vector(), b)
    eps = np.linalg.norm(np.array(du.vector()), ord=np.Inf)
    print ('Norm:', eps)
    u.vector()[:] = u_k.vector() + omega*du.vector()
    u_k.assign(u)
#    plotting_normal(u_k, 'Du')
    
    elecint = conditional(gt(u_k,0.0),u_k  , 1E-8)
    # if u_n > 0.0 elecint == u_n*r^2 
    # else          elecint == 0.0
    intn = 4*math.pi*float(assemble((elecint)*dx))
    print("Electron maximum:",intn)
     
gr = project(u_k.dx(0),V)
nr = project(Z/(4.0*pi)*gr.dx(0)/r,V)

plotting_psi(u_k, "density - PSI")
#plotting_normal(gr, " GR" )
#plotting_normal(nr, " NR")


"""
plt.figure()
plt.title('Thomas-Fermi screening function Psi(x) for various slopes of x_0')
plt.xlabel("Radial coordinate")
plt.ylabel('Psi')
plt.grid()
plot(A1)
plot(A4)
plot(A8)
plot(A20)
plt.show()
"""