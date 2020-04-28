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


plt.close('all')

"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Plot function
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""

def plotting_normal(u,title):
    rplot = (mesh.coordinates())
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
    rplot = (mesh.coordinates())
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
     
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating the mesh + Function Spaces + defining type of element
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------""" 
# Create mesh and define function space
start_x = 0.3
end_x = 20
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

#Element H
Z = Constant(1) # Hydrogen
N = Z # Neutral

#Element Ne
#Z = Constant(10) # Neon
#N = Z # Neutral

#Element Kr
#Z = Constant(36) # Krypton
#N = Z # Neutral 
"""---------------------------------------------------------------------------"""
## ------ Tweaking values -------
neg_correction = 0.3
omega = 1
mu = 0

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
# External potential

Ex = -Z/r

#------ Initial density ------

#n_i = Expression('a*exp(-b*pow(x[0], 2))', degree = 2, a = 1, b=1.0)
#n_i = Expression('pow(x[0],2)', degree = 2)
n_i = Constant(1)

u_n = interpolate(n_i,V)

#------------Checking amount of electrons ----------------------

intn = float(assemble((u_n)*dx(mesh)))
print("[Initial density] Number of electrons before adjustment:"+str(intn))
u_n.vector()[:] = u_n.vector()[:]*N/intn  
intn = float(assemble((u_n)*dx(mesh)))
print("[Initial density] Number of electrons at start after adjustment:",intn)

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

bcs_du = []


###-------try getting the functionals to work properly
CF=(3.0/10.0)*(3.0*math.pi**2)**(2.0/3.0)
CX = 3.0/4.0*(3.0/math.pi)**(1.0/3.0)



eps = 1
iters = 0
maxiter = 1500
minimal_error = 1E-9

while eps > minimal_error and iters < maxiter:
    iters += 1 
    
    # Put the initial [density and v_h] in u_k
    (v_hk, u_nk) = split(u_k)
    #plotting_normal(u_nk, "density beginning loop") #for verifying with previous results
    plotting_sqrt(u_nk, "density beginning loop")   #For verifying with 'accurate solution...' paper
    
    #---- Setting up functionals -------------------
    
    TF = (5.0/3.0)*CF*u_nk**(2.0/3.0)*pr
    DIRAC = (-4.0/3.0)*CX*pow(u_nk,(1.0/3.0))*pr
    WEIZSACKER = ((1.0/8.0*(dot(u_nk.dx(0),u_nk.dx(0))/(u_nk*u_nk))*pr+(1.0/4.0*(dot(u_nk.dx(0),pr.dx(0)))/u_nk)))    
    funcpots = TF + DIRAC + WEIZSACKER       
    
    #---- Solving v_h and u_n ----------------------
    # rotational transformation of nabla^2 v_h = -4 pi n(r)
    F = - v_hk.dx(0)*qr.dx(0)*dx    \
        + (2/r)*v_hk.dx(0)*qr*dx  \
        + 4*math.pi*u_nk*qr*dx              
         
    # Second coupled equation: Ts[n] + Exc[n] + Vext(r) - mu = 0
    F = F + Ex*pr*dx + funcpots*dx + v_hk*pr*dx - Constant(mu)*pr*dx


    #Calculate Jacobian
    J = derivative(F, u_k, du_trial)
    
    #Assemble system
    A, b = assemble_system(J, -F, bcs)
    
    rvec = mesh.coordinates()
    nvec = np.array([v_hk(v) for v in rvec])
    minval = nvec.min()
    print("v_hk minimum:",minval)

    rvec = mesh.coordinates()
    nvec = np.array([u_nk(v) for v in rvec])
    minval = nvec.min()
    print("u_nk minimum:",minval) 
      
    solve(A, du.vector(), b)
    
    # Conserve memory by reusing vectors for v_h and u_n to keep dv_h and du_n
    dv_h = v_h                  #step to transfer type info to dv_h
    assign(dv_h, du.sub(0))     #step to transfer values to dv_h
    v_h = None                  #empty v_h
    
    du_n = u_n                  #step to transfer type info to du_n
    assign(du_n, du.sub(1))     #step to transfer values to du_n
    u_n = None                  #empty u_n

    #plotting_normal(du_n, "du_n (Step we want density to take)") #for verifying with previous results
    plotting_sqrt(du_n,"du_n (step we want the density to take)") #For verifying with 'accurate solution...' paper

    
    #---- Calculate the Error -----------------------
    avg = sum(du.vector())/len(du.vector())
    eps = np.linalg.norm(du.vector()-avg, ord=np.Inf)
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

    #plotting_normal(u_n,"density post solver / pre neg fix")
    
    
    #---- Ad hoc negative density fix -------
    minval = u_n.vector().min()
    print("Going to add:", minval+neg_correction)
    if minval <= 0.0:
        u_n.vector()[:]= u_n.vector()[:]-minval+neg_correction
        intn = float(assemble((u_n)*dx(mesh)))
        print("Number of electrons before correction:",intn)
        
        u_n.vector()[:] = u_n.vector()[:]*N/intn    
        intn = float(assemble((u_n)*dx(mesh)))            
        print("Number of electrons after correction:",intn)

    assign(u_k.sub(1),u_n)
    minval = u_k.sub(1).vector().min()
    
    print("Minval after correction:",minval)
    print('Check end of loop, iteration: ', iters,' Error: ', eps)
    
    #plotting_normal(u_k.sub(1), "Density post neg fix" )
    
