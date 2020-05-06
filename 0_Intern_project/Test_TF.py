#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 09:37:06 2020

@author: rick
"""
from __future__ import print_function
import math
import numpy as np
from dolfin import *
import matplotlib as mpl
from matplotlib import pyplot as plt


# Create mesh and define function space
start_x = 0.0
end_x = 5
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
n_i = Constant(10)


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

alpha = 4.0/1.0*(2.0*Z/(9.0*pi**2))**(1.0/3.0)

# Define variational problem
u_i = interpolate(Constant(0), V)  # previous (known) u
u = TrialFunction(V)
v = TestFunction(V)
r = SpatialCoordinate(mesh)[0]
a = -u.dx(0)*v.dx(0)*dx 
L = alpha**(3.0/2.0)*(1.0/(sqrt(r)))*(sqrt(u_i)**3)*v*dx
A, b = assemble_system(a, L, bcs)
u_k = Function(V)
solve(A, u_k.vector(), b)



"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Plot function
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""

def plotting_psi(u,title):
    rplot = mesh.coordinates()
    x = rplot*Alpha
    y = [u(v)*(2/3)*r*(3*math.pi**2/(2*np.sqrt(2)))**(2/3)*a_0/Z  for v in rplot]
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Radial coordinate")
    plt.ylabel(title)
    plt.grid()
    plt.plot(x,y)
    plt.show()
    return     

def plotting_normal(u,title):
    rplot = mesh.coordinates()
    x = rplot
    y = [u(v) for v in rplot]
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Radial coordinate")
    plt.ylabel(title)
    plt.grid()
    plt.plot(x,y)
    plt.show()
    return 


def plotting_sqrt(u,title):
    rplot = mesh.coordinates()
    x = np.sqrt(rplot)
    y = [v*sqrt(u(v)) for v in rplot] 
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Radial coordinate")
    plt.ylabel(title)
    plt.grid()
    plt.plot(x,y)
    plt.show()
    return 

plotting_normal(u_, "Normal plot density")
plotting_sqrt(u_, "SQRT plot density")
#plotting_psi(u_, "Normal plot density")