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
Int = 0
Potential = Ex +Int

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

n_i = Constant(1)

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Defining and solving the variational problem
                    defining trial and test functions
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
#--Rewriting variables for overview--
#x = sqrt(r)
#y= r*sqrt(u)
#Q = r*(Potential)

u_i = TrialFunction(V)
v = TestFunction(V)

#Solving for internal potential with initial density
F1 = (r*(Ex+u_i)).dx(0)*v.dx(0)*dx                                  \
    - 1/sqrt(r)*(r*(Ex+u_i)).dx(0)*v.dx(0)*dx                       \
    - 16*math.pi*r*sqrt(n_i)*v*dx                                   \
    + (r*sqrt(n_i)).dx(0)*v.dx(0)*dx                                \
    - r/sqrt(r)*sqrt(n_i)*v*dx                                      \
    - (5*C1)/(3*C3)*((r*sqrt(n_i))**(7/3))/(sqrt(r)**(5/3))*v*dx    \
    + 4/3*sqrt(r)**(7/3)*(r*sqrt(n_i))**(5/4)*v*dx                  \
    - 1/C3*r*(mu-Ex-u_i)*v*dx 

u_i = Function(V)
F1 = action(F1, u_i)
J1 = derivative(F1, u_i)

#set_boundary(0,1,1E-14,'0','0', V)

problem = NonlinearVariationalProblem(F1, u_i, bcs, J1)
solver = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-9
prm['newton_solver']['relative_tolerance'] = 1E-6
prm['newton_solver']['maximum_iterations'] = 1000
prm['newton_solver']['relaxation_parameter'] = 1.0
solver.solve()

u_i = u_i +10
plotting_solve_result(u_i, False)
"""-----------------------------------------------------------------------"""

#Solving for density with input internal potential from previous solve
u_n = TrialFunction(V)
v = TestFunction(V)

F2 = ((r*(Ex+u_i)).dx(0)*v.dx(0)*dx                                  \
    - 1/sqrt(r)*(r*(Ex+u_i)).dx(0)*v.dx(0)*dx                       \
    - 16*math.pi*r*sqrt(u_n)*v*dx                                   \
    + (r*sqrt(u_n)).dx(0)*v.dx(0)*dx                                \
    - r/sqrt(r)*sqrt(u_n)*v*dx                                      \
    - (5*C1)/(3*C3)*((r*sqrt(u_n))**(7/3))/(sqrt(r)**(5/3))*v*dx    \
    + 4/3*sqrt(r)**(7/3)*(r*sqrt(u_n))**(5/4)*v*dx                  \
    - 1/C3*r*(mu-Ex-u_i)*v*dx )

u_n = Function(V)
F2 = action(F2, u_n)
J2 = derivative(F2, u_n)

#set_boundary(0,1,1E-14,'0','0',V)

problem = NonlinearVariationalProblem(F2, u_n, bcs, J2)
solver = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-9
prm['newton_solver']['relative_tolerance'] = 1E-6
prm['newton_solver']['maximum_iterations'] = 10000
prm['newton_solver']['relaxation_parameter'] = 1.0

solver.solve()

plotting_solve_result(u_n, True)

gr = project(u_n.dx(0),V)
plotting_solve_result(gr, False)

nr = project(Z/(4.0*pi)*gr.dx(0)/r,V)
plotting_solve_result(nr, False)


"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Saving VTKfile for post processing in ParaView
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
#Save solution to file in VTK format
vtkfile = File('VTKfiles/radial_atom_solver_TF.pvd')
vtkfile << u_n

"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Computing the error 
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
#