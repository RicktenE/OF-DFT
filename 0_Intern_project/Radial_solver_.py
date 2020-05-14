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

def plotting_psi(u,title):
    a_0 = 1 # Hatree units
    Alpha_ = (4/a_0)*(2*Z/(9*pi**2)**(1/3))
    
    pylab.clf()
    rplot = mesh.coordinates()
    x = rplot*Alpha_
    y = [u(v)**(2/3)*v*(3*math.pi**2/(2*np.sqrt(2)))**(2/3)  for v in rplot]
    
    #x = numpy.logspace(-5,2,100)
    y = [u(v)**(2/3)*v*(3*math.pi**2/(2*np.sqrt(2)))**(2/3)  for v in rplot]
    pylab.plot(x,y,'bx-')
    pylab.title(title)
    pylab.pause(0.001)
    pylab.xlabel("Alpha * R")
    pylab.ylabel("Psi")

    return     

def plotting_normal(u,title):
    pylab.clf()
    
    rplot = mesh.coordinates()
    x = rplot
    #x = numpy.logspace(-5,2,100)
    y = [u(v) for v in rplot]

    pylab.plot(x,y,'bx-')
    pylab.title(title)
    pylab.pause(1)
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
    
    pylab.plot(x,y,'bx-')
    pylab.title(title)
    pylab.pause(0.001)
    pylab.xlabel("SQRT(R)")
    pylab.ylabel("R * SQRT(density")
    
    return 
     
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating the mesh + Function Spaces + defining type of element
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------""" 


# Create mesh and define function space
start_x = 0.1
end_x = 9.0
amount_vertices = 200
mesh = IntervalMesh(amount_vertices,start_x, end_x) # Splits up the interval [0,1] in (n) elements 

#print("MES COORDS",np.shape(mesh.coordinates()))

#Creation of Function Space
P1 = FiniteElement("P", mesh.ufl_cell(), 2)
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
n_L = Expression('0', degree=1)         
bc_L = DirichletBC(V, n_L, boundary_L)  
    
#Defining expression on right boundary
n_R = Expression('0', degree=1)
bc_R = DirichletBC(V, n_R, boundary_R) 
    
#collecting the left and right boundary in a list for the solver to read
#bcs = [bc_L, bc_R]
bcs = []
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Defning external potential v[r] and Initial density n_1[r]
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
# External potential
Ex = -Z/r

#------ Initial density ------
n_i = Expression('a*exp(-b*pow(x[0], 2))', degree = 2, a = 1, b=0.05)
#n_i = Constant(1)

u_n = interpolate(n_i, V)

#plotting_normal(u_n,"Initial density--pre correction")

u = TrialFunction(V)
v = TestFunction(V)
a = -u.dx(0)*v.dx(0)*dx 
L = sqrt(n_i)**3*v*dx
A,  b = assemble_system(a, L, bcs)
u_k = Function(V)
solve(A, u_n.vector(), b)

#plotting_normal(u_n,"Initial density-- Post initial solve --Pre correction")

#------------Checking amount of electrons ----------------------

intn = float(assemble((u_n)*dx(mesh)))
print("[Initial density] Number of electrons before adjustment:"+str(intn))
u_n.vector()[:] = u_n.vector()[:]*N/intn  
intn = float(assemble((u_n)*dx(mesh)))
print("[Initial density] Number of electrons at start after adjustment:",intn)
plotting_normal(u_n,"Initial density--Post correction -- Begin loop")
#print(type(u_n))
"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Defining and solving the variational problem
                    defining trial and test functions
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
v_h = interpolate(Constant(-1), V)

mixed_test_functions = TestFunction(W)
(qr, pr) = split(mixed_test_functions)

du_trial = TrialFunction(W)        
du = Function(W)

u_k = Function(W)
assign(u_k.sub(0), v_h)
assign(u_k.sub(1), u_n)

nlast = Function(V)

#redefine boundary conditions for loop iteration
bc_L_du = DirichletBC(V, Constant(0), boundary_L)
bc_R_du = DirichletBC(V, Constant(0), boundary_R)
#bcs_du = [bc_L_du, bc_R_du]

bcs_du = []
"""---------------------------------------------------------------------------"""
## ------ Tweaking values -------
neg_correction = 0.1
omega = 1
mu = 100

eps = 1
iters = 0
maxiter = 1000
minimal_error = 1E-10

while eps > minimal_error and iters < maxiter:
    iters += 1 
    
    # Put the initial [density and v_h] in u_k
    (v_hk, u_nk) = split(u_k)
    
    #plotting_psi(nlast, "Psi begin of loop ")
    plotting_sqrt(u_nk, "density pre solver")   #For verifying with 'accurate solution...' paper
    #plotting_normal(u_nk,"density pre solver")
    #---- Setting up functionals -------------------
    TF = (5.0/3.0)*CF*u_nk**(2.0/3.0)*pr
    DIRAC = (-4.0/3.0)*CX*pow(u_nk,(1.0/3.0))*pr
    WEIZSACKER = (1.0/8.0*(dot(grad(u_nk),grad(u_nk))/		 (u_nk**2)*pr+(1.0/4.0*(dot(grad(u_nk),grad(pr)))/u_nk)))
    funcpots = 0
    funcpots = TF \
		+ WEIZSACKER \
       + DIRAC 
      
    
    #---- Solving v_h and u_n ----------------------
    # rotational transformation of nabla^2 v_h = -4 pi n(r)
    F = - v_hk.dx(0)*qr.dx(0)*r/2*dx    \
    + v_hk.dx(0)*qr*dx  \
    + 4*math.pi*u_nk*qr*r/2*dx              
         
    # Second coupled equation: Ts[n] + Exc[n] + Vext(r) - mu = 0
    F = F + funcpots*dx \
    + v_hk*pr*dx \
    + Ex*pr*dx +\
    - Constant(mu)*pr*dx

    #Calculate Jacobian
    J = derivative(F, u_k, du_trial)
    
    #Assemble system
    A, b = assemble_system(J, -F, bcs_du)
    
    """
    rvec = mesh.coordinates()
    nvec = np.array([v_hk(v) for v in rvec])
    minval = nvec.min()
    print("v_hk minimum:",minval)

    rvec = mesh.coordinates()
    nvec = np.array([u_nk(v) for v in rvec])
    minval = nvec.min()
    print("u_nk minimum:",minval) 
    """
    solve(A, du.vector(), b)
    
    # Conserve memory by reusing vectors for v_h and u_n to keep dv_h and du_n
    dv_h = v_h                  #step to transfer type info to dv_h
    assign(dv_h, du.sub(0))     #step to transfer values to dv_h
    v_h = None                  #empty v_h
    
    du_n = u_n                  #step to transfer type info to du_n
    assign(du_n, du.sub(1))     #step to transfer values to du_n
    u_n = None                  #empty u_n

    #plotting_normal(du_n, "du_n (Step for density)") #for verifying with TF results
    #plotting_sqrt(du_n,"du_n (step for the density to take)") #For verifying with TFDW
    #plotting_psi(du_n,"du_n (step for psi to take)") #For verifying with TFDW
    
    #---- Calculate the Error -----------------------
    avg = sum(du_n.vector())/len(du_n.vector())
    eps = np.linalg.norm(du_n.vector()-avg, ord=np.Inf)
    if math.isnan(eps):
            raise Exception("Residual error is NaN")
    print("EPS is:",eps)
        
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

     #---- Ad hoc negative density fix -------
    minval = u_n.vector().min()
    print("Going to add:", minval+neg_correction)
    if minval <= 0.0:
        u_n.vector()[:]= u_n.vector()[:]-minval+neg_correction
        intn = float(assemble((u_n)*dx(mesh)))
        print("Number of electrons before correction:",intn)
        
        assign(u_k.sub(1),u_n)
        u_n.vector()[:] = u_n.vector()[:]*N/intn    
        intn = float(assemble((u_n)*dx(mesh)))            
        print("Number of electrons after correction:",intn)

    minval = u_k.sub(1).vector().min()
    
    #print("Minval after correction:",minval)
    print('Check end of loop, iteration: ', iters)

#plotting_normal(nlast, "Final Density normal plot ")
#plotting_sqrt(nlast, "Final Density SQRT plot")
#plotting_psi(nlast, "Final Psi ")