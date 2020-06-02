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

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Plot function
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
  

def plotting(u,title):
    pylab.clf()
    
    rplot = mesh.coordinates()
    x = rplot
    y = [u(v) for v in rplot]

    pylab.plot(x,y,'bx-')
    pylab.title(title)
    #pylab.xlim(0, 8)
    #pylab.ylim(0, 1)
    pylab.grid
    pylab.xlabel("X")
    pylab.ylabel("Psi")
    
    return 
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating the mesh 
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------""" 
# Create mesh and define function space
start_x = 0.01
end_x = 3.0
amount_vertices = 200
mesh = IntervalMesh(amount_vertices,start_x, end_x) # Splits up the interval [x_start,x_end] in (n) elements 
V = FunctionSpace(mesh, 'P', 1) # P stands for lagrangian elemnts, number stands for degree
x = SpatialCoordinate(mesh)[0] # r are the x coordinates. 

'''----------------------constants-------------------------------'''

a_0 = 1 # if all natural constants are 1
Z = 1 # if all natural constants are 1

"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Defning external potential v[r] and Initial density n_1[r]
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""

psi_i = Expression('1.0-x[0]/zp', degree=2,zp=end_x)

"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating and defining the boundary conditions
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
#Defining the tolerance on the boundaries 
tol = 1E-14

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Defining and solving the variational problem
                    defining trial and test functions
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
u = TrialFunction(V)
v = TestFunction(V)
psi_k = interpolate(psi_i, V)  # previous (known) u

def boundary_L(x, on_boundary):
    return on_boundary and near(x[0], start_x, tol)
def boundary_R(x, on_boundary):
    return on_boundary and near(x[0], end_x, tol)
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
while eps > tol and iter < maxiter:
    iter += 1
    
    #Redifine trial function and derivative of function
    F  = -psi_k.dx(0)*v.dx(0)*dx - psi_k**(3/2)/x**(1/2)*v*dx + psi_k.dx(0)*v*ds(0)
    J = derivative(F, psi_k, du_)
    plotting(psi_k,'Density - PSI')

    A, b = assemble_system(J, -F, bcs_du)
    solve(A, du.vector(), b)
    eps = np.linalg.norm(np.array(du.vector()), ord=np.Inf)
    print ('Norm:', eps)
    u.vector()[:] = psi_k.vector() + omega*du.vector()
    psi_k.assign(u)
#    plotting_normal(psi_k, 'Du')
     
gr = project(psi_k.dx(0),V)
#nr = project(Z/(4.0*pi)*gr.dx(0)/x,V)

print("q=",-gr(end_x)*end_x)

plotting(psi_k, "density - PSI")
pylab.waitforbuttonpress()
plotting(gr, " GR" )
#plotting(gr, " NR")
plt.show()

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
