"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  -Laplace(u) = f    in the unit square
            u = u_D  on the boundary

  u_D = 1 + x^2 + 2y^2
    f = -6
"""

from __future__ import print_function
from fenics import UnitSquareMesh,plot, FunctionSpace, near, Function, errornorm, Expression, DirichletBC, TrialFunction, TestFunction, dot, grad, Constant, dx, ds, solve, File 
import matplotlib.pyplot as plt


# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary condition
#u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

tol = 1E-14

#Defining the left boundary
def boundary_L(x, on_boundary):
    return on_boundary and near(x[0], 0, tol)

#Defining the right boundary
def boundary_R(x, on_boundary):
    return on_boundary and near(x[0], 1, tol)

#defining expression on left boundary
u_L = Expression('1 + 2*x[1]*x[1]', degree=2) 
#Define left boundary
bc_L = DirichletBC(V, u_L, boundary_L) 
#defining expression on right boundary
u_R = Expression('2 + 2*x[1]*x[1]', degree=2) 
#Define right boundary
bc_R = DirichletBC(V, u_R, boundary_R) 

#collecting the left and right boundary in a list for the solver to read
bcs = [bc_L, bc_R] 

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

f = Constant(-6)
a = dot(grad(u), grad(v))*dx
g = Expression('-4*x[1]', degree=1) 
L = f*v*dx - g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bcs)

# Plot solution and mesh
#plt.figure()

plot(u)
plot(mesh)


# Save solution to file in VTK format
vtkfile = File('poisson/solution.pvd')
vtkfile << u

# Compute error in L2 norm
error_L2 = errornorm(u_L, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_L.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)


plt.show()

  
