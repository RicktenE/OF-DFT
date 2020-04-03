#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

"""
FEniCS tutorial demo program: Convection-diffusion-reaction for a system
describing the concentration of three species A, B, C undergoing a simple
first-order reaction A + B --> C with first-order decay of C. The velocity
is given by the flow field w from the demo navier_stokes_cylinder.py.

  u_1' + w . nabla(u_1) - div(eps*grad(u_1)) = f_1 - K*u_1*u_2
  u_2' + w . nabla(u_2) - div(eps*grad(u_2)) = f_2 - K*u_1*u_2
  u_3' + w . nabla(u_3) - div(eps*grad(u_3)) = f_3 + K*u_1*u_2 - K*u_3
"""

from fenics import *
from mshr import *

T = 5.0            # final time
num_steps = 500    # number of time steps
dt = T / num_steps # time step size
eps = 0.01         # diffusion coefficient
K = 10.0           # reaction rate

# Create mesh 
channel = Rectangle(Point(0, 0), Point(2.2, 0.41)) 
cylinder = Circle(Point(0.2, 0.2), 0.05) 
domain = channel - cylinder 
mesh = generate_mesh(domain, 64)

# Define function space for velocity
W = FiniteElementFunctionSpace(mesh, 'P', 2)

# Define function space for system of concentrations
P1 = FiniteElement('P', triangle, 1)
element = MixedElement([P1, P1, P1])
V = FunctionSpace(mesh, element)

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
bc_L = DirichletBC(W, n_L, boundary_L)  
    
#Defining expression on right boundary
n_R = Expression('0', degree=1)
bc_R = DirichletBC(W, n_R, boundary_R) 
    
#collecting the left and right boundary in a list for the solver to read
bcs = [bc_L, bc_R]


# Define test functions
v_1, v_2, v_3 = TestFunctions(V)

# Define functions for velocity and concentrations
w = Function(W)
u = Function(V)
u_n = Function(V)

# Split system functions to access components
u_1, u_2, u_3 = split(u)
u_n1, u_n2, u_n3 = split(u_n)

# Define source terms
f_1 = Expression('pow(x[0]-0.1,2)+pow(x[1]-0.1,2)<0.05*0.05 ? 0.1 : 0',
                 degree=1)
f_2 = Expression('pow(x[0]-0.1,2)+pow(x[1]-0.3,2)<0.05*0.05 ? 0.1 : 0',
                 degree=1)
f_3 = Constant(0)

# Define expressions used in variational forms
k = Constant(dt)
K = Constant(K)
eps = Constant(eps)

# Define variational problem
F = ((u_1 - u_n1) / k)*v_1*dx + dot(w, grad(u_1))*v_1*dx \
  + eps*dot(grad(u_1), grad(v_1))*dx + K*u_1*u_2*v_1*dx  \
  + ((u_2 - u_n2) / k)*v_2*dx + dot(w, grad(u_2))*v_2*dx \
  + eps*dot(grad(u_2), grad(v_2))*dx + K*u_1*u_2*v_2*dx  \
  + ((u_3 - u_n3) / k)*v_3*dx + dot(w, grad(u_3))*v_3*dx \
  + eps*dot(grad(u_3), grad(v_3))*dx - K*u_1*u_2*v_3*dx + K*u_3*v_3*dx \
  - f_1*v_1*dx - f_2*v_2*dx - f_3*v_3*dx


# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Solve variational problem for time step
    solve(F == 0, u)


    # Update previous solution
    u_n.assign(u)
    print('loop', n)