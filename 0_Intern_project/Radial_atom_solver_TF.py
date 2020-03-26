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
from fenics import *
import matplotlib as mpl
from matplotlib import pyplot as plt


plt.close('all')
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating the mesh 
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------""" 
# Create mesh and define function space
start_x = 0
end_x = 2
amount_vertices = 100
mesh = IntervalMesh(amount_vertices,start_x, end_x) # Splits up the interval [0,1] in (n) elements 
V = FunctionSpace(mesh, 'P', 1) # P stands for lagrangian elemnts, number stands for degree
r = SpatialCoordinate(mesh)[0] # r are the x coordinates. 

'''----------------------constants-------------------------------'''

# e = 1.60217662E-19
# h_bar = 1.0545718E-34
# m_e = 9.109E-31
# Z=1.602E-19
# a_0 = sqrt(m_e/h_bar**2)**3*e**2


a_0 = 1 # if all natural constants are 1
Z = 1 # if all natural constants are 1
e = 1 # if all natural constants are 1
C1 = 8*sqrt(2)/(3*math.pi)
C2 = a_0**(3/2)/e

mu = 0

"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Defning external potential v[r] and Initial density n_1[r]
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
#External potential v_e[r] is analytically described for atoms 
Ex = -Z/r

#Initial density n_1[r]
a=1/sqrt(2*pi)
b=1
m_i = Constant(0)
#n_i = a*exp(pow((-b*(r)), 2))
#n_i = Expression('a*exp(pow((-b*(x[0])), 2))', degree =1, a =1/sqrt(2*pi), b=1)


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
n_L = Expression('0', degree=1)         #Eq. 5,189
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
#Thomas Fermi Energy Functional
# C_kin = 3/10*pow((3*pow(math.pi, 2)),(2/3))
# TF_KE = C_kin*pow(u,(5/3))
# #Thomas Fermi Energy functional derived towards u
# TF_KE_der = 0.5*pow((3*pow(math.pi, 2)),(2/3))*pow(u,(3/2))


#a=1/sqrt(2*pi)
#b=1
#u_form = a*exp(-b*r**2)
#u_form = Expression('a*exp(pow((-b*(x[0])), 2))', degree =1, a =1/sqrt(2*pi), b=1)
#u = interpolate(u_form, V)
u = TrialFunction(V)
v = TestFunction(V)
#F = -u.dx(0)*v.dx(0)*dx -(1.0/(sqrt(r)))*u**(3/2)*v*dx #Basic variational problem to show functionality
F  = u.dx(0)*v.dx(0)*dx - C1*C2*(mu-Ex-u)**(2/2)*v*dx
u_l = Function(V)
F = action(F, u_l)
J = derivative(F, u_l)

problem = NonlinearVariationalProblem(F, u_l, bcs, J)
solver = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-9
prm['newton_solver']['relative_tolerance'] = 1E-6
prm['newton_solver']['maximum_iterations'] = 1000
prm['newton_solver']['relaxation_parameter'] = 1.0

   
solver.solve()

gr = project(u_l.dx(0),V)
nr = project(Z/(4.0*pi)*gr.dx(0)/r,V)

"""
#redefine boundary conditions for loop iteration
bc_L_du = DirichletBC(V, Constant(0), boundary_L)
bc_R_du = DirichletBC(V, Constant(0), boundary_R)
bcs_du = [bc_L_du, bc_R_du]

#Coupled diferential equations? 

#Redifine trial function and derivative of function
du_ = TrialFunction(V)
F  = -u_k.dx(0)*v.dx(0)*dx - C1*C2*pow(sqrt(mu-Ex-u_k),3)*v*dx
J = derivative(F, u_k, du_)

# Start second solve
du = Function(V)
u  = Function(V)  # u = u_k + omega*du
omega = 1.0       # relaxation parameter
eps = 1.0

#Initiate loop for convergence on value for u
iter = 0
maxiter = 50
while eps > tol and iter < maxiter:
    iter += 1
    A, b = assemble_system(J, -F, bcs_du)
    solve(A, du.vector(), b)
    eps = np.linalg.norm(np.array(du.vector()), ord=np.Inf)
    print ('Norm:', eps)
    u.vector()[:] = u_k.vector() + omega*du.vector()
    u_k.assign(u)
    
gr = project(u_l.dx(0),V)
nr = project(Z/(4.0*pi)*gr.dx(0)/r,V)
"""

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Saving VTKfile for post processing in ParaView
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
# Save solution to file in VTK format
vtkfile = File('VTKfiles/radial_atom_solver_TF.pvd')
vtkfile << u_l

"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Computing the error 
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
# There is no analytical solution to check the error

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Plot solution and mesh
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""


# plt.figure(1)
# plt.title("gr")
# plt.xlabel("Radial coordinate")
# plt.ylabel("projection of derivative of n towards r")
# plt.grid()
# plot(gr)

# plt.figure(2)
# plt.title("nr")
# plt.xlabel("Radial coordinate")
# plt.ylabel("projection of Z/(4.0*pi)*gr.dx(0)/r ")
# plt.grid()
# plot(nr)


plt.figure()
plt.title("Last calculated density u_l")
plt.xlabel("Radial coordinate")
#mpl.scale.LogScale(r)
plt.ylabel("Density [n]")
plt.grid()
plot(u_l)


#plot(mesh)

# show the plots
plt.show()