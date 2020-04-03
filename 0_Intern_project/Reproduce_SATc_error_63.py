
#---Minimal running example to reproduce the error --- 

from __future__ import print_function
import math
import numpy as np
from dolfin import *


"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating the mesh + Function Spaces + defining type of chemical element
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------""" 
# Create mesh and define function space
start_x = 0
end_x = 2
amount_vertices = 100
mesh = IntervalMesh(amount_vertices,start_x, end_x) # Splits up the interval [0,1] in (n) elements 

#Creation of Function Space
P1 = FiniteElement("P", mesh.ufl_cell(), 2)
element = MixedElement([P1,P1])

V = FunctionSpace(mesh, 'P', 2) # P stands for lagrangian elemnts, number stands for degree
W = FunctionSpace(mesh, element)

#Define radial coordinates based on mesh
r = SpatialCoordinate(mesh)[0] # r are the x coordinates. 

#Chemical element Kr
Z = Constant(36) # Krypton
N = Z # Neutral 

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

#Initial density n_i[r]
n_i = Constant(0)

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Defining and solving the variational problem
                    defining trial and test functions
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""

u_n = interpolate(n_i, V)
v_h = interpolate(Constant(-1), V)

mixed_test_functions = TestFunction(W)
(vr, pr) = split(mixed_test_functions)

du_trial = TrialFunction(W)        
du = Function(W)

u_k = Function(W)
assign(u_k.sub(0), v_h)
assign(u_k.sub(1), u_n)

(v_hk, u_nk) = split(u_k)
#--Rewriting variables for overview--
x = sqrt(r)
y= r*sqrt(u_nk)
Q = r*(Ex+v_hk)

bcs_du = []
eps = 1
iters = 0
maxiter = 5000
eps2 = 2

while eps > tol and iters < maxiter:
    iters += 1 
    (v_hk, u_nk) = split(u_k)
    
    #First coupled equation: Q'' = 1/x*Q' +16pi*y^2
    F = -Q.dx(0)*vr.dx(0)*dx                                \
        + 1/x*Q.dx(0)*vr.dx(0)*dx                           \
        + 16*math.pi*y*vr*dx  
    
    # Second coupled equation y'' = ... y ... x ... Q
    F = F - y.dx(0)*pr.dx(0)*dx                             \
        + y/x*pr*dx                                         \
        + (5*C1)/(3*C3)*(y)**(7/3)/(x)**(5/3)*pr*dx         \
        - 4/3*(x)**(7/3)*(y)**(5/4)*pr*dx                   \
        + 1/C3*Q*pr*dx  
    
    #Calculate Jacobian
    J = derivative(F, u_k, du_trial)
 
    #Assemble system
    A, b = assemble_system(J, -F, bcs_du)
    print('check 3')
    solve(A, du.vector(), b)
