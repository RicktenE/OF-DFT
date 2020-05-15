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
    pylab.plot(x,y,'kx-')
    pylab.title(title)
    pylab.pause(0.1)
    pylab.xlabel("Alpha * R")
    pylab.ylabel("Psi")

    return     

def plotting_normal(u,title):
    pylab.clf()
    
    rplot = mesh.coordinates()
    x = rplot
    #x = numpy.logspace(-5,2,100)
    y = [u(v) for v in rplot]

    pylab.plot(x,y,'kx-')
    pylab.title(title)
    pylab.pause(0.1)
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
    
    pylab.plot(x,y,'kx-')
    pylab.title(title)
    pylab.pause(0.05)
    pylab.xlabel("SQRT(R)")
    pylab.ylabel("R * SQRT(density")
    
    return 
     
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating the mesh + Function Spaces + defining type of element
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------""" 


# Create mesh and define function space
# =============================================================================
# rs = np.arange(0.1,5.0,0.1)
# rs = np.array((rs**2))
# =============================================================================
rs = np.arange(0.1,20.0,0.1)
#rs = np.array((rs**2))

start_x = rs[0]
end_x = rs[-1]
amount_vertices = len(rs)
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
                    Defning external potential v[r] 
                    Initial density n_1[r]
                    Hartree potential v_h
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
# External potential
Ex = -Z/r

#########------ Initial density ------##########
n_i = Expression('a*exp(-b*pow(x[0], 2))', degree = 2, a = 1, b=0.05)
#n_i = Expression('rho0*exp(-fac*Z*x[0])', degree=2, Z=Z,fac=1.8,rho0=5832)
#n_i = Constant(1)

u_n = interpolate(n_i, V)

##plotting_sqrt(u_n, "PRE first solve")

u = TrialFunction(V)
v = TestFunction(V)
a = u.dx(0)*v.dx(0)*dx 
L = u_n**(3/2)*v*dx
#L = sqrt(u_n)*3*v*dx
#L = u_n*v*dx

A,  b = assemble_system(a, L, bcs)

solve(A, u_n.vector(), b)

##plotting_sqrt(u_n, "POST first solve")

#------------Checking amount of electrons ----------------------

intn = float(assemble((u_n)*dx(mesh)))
print("[Initial density] Number of electrons before adjustment:", intn)
u_n.vector()[:] = u_n.vector()[:]*N/intn  
intn = float(assemble((u_n)*dx(mesh)))
print("[Initial density] Number of electrons at start after adjustment:",intn)


nvec = u_n.vector()
minval = nvec.min()
print("minval PRE neg fix:",minval)
print(" PRE nvec values", nvec.get_local())
plotting_sqrt(u_n,"PRE neg fix ")
# =============================================================================
# #nvec[nvec<0.0]=0.0  #puts index -1 to 0 ?
# print("nvec values", nvec.get_local())
# 
# nvec[nvec.get_local()[nvec<20]] = 0.0
# nvec[nvec.get_local()[-2]] = 0.0
# nvec[nvec.get_local()[-3]] = 0.0
# nvec[nvec.get_local()[-4]] = 0.0
# =============================================================================


nvec[nvec<0.0] = 0.0   #puts index -1 to 0 ?
#nvec[rs[0]] = 0.0


nvec = u_n.vector()
minval = nvec.min()
print("minval POST neg fix",minval)
print("POST nvec values", nvec.get_local())
plotting_sqrt(u_n,"POST neg fix ")
#######----- Initializing boundary conditions on Hartree potential ---###

v_h = interpolate(Constant(-1), V)

# =============================================================================
# u = TrialFunction(V)
# v = TestFunction(V)
# a = u.dx(0)*v.dx(0)*dx 
# L = v_h*v*dx
# #L = sqrt(n_i)*v*dx
# #L = n_i*v*dx
# A,  b = assemble_system(a, L, bcs)
# 
# plotting_sqrt(v_h, "VH  - PRE first solve")
# 
# solve(A, v_h.vector(), b)
# 
# plotting_sqrt(v_h, "VH  - POST first solve")
# =============================================================================
"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
                    Defining and solving the variational problem
                    defining trial and test functions
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""

mixed_test_functions = TestFunction(W)
(qr, pr) = split(mixed_test_functions)

du_trial = TrialFunction(W)        
du = Function(W)

u_k = Function(W)
assign(u_k.sub(0), v_h)
assign(u_k.sub(1), u_n)
#plotting_sqrt(u_k.sub(1), " U-k(1)")
nlast = Function(V)

# =============================================================================
# #redefine boundary conditions for loop iteration
# bc_L_du = DirichletBC(V, Constant(0), boundary_L)
# bc_R_du = DirichletBC(V, Constant(0), boundary_R)
# #bcs_du = [bc_L_du, bc_R_du]
# =============================================================================

bcs_du = []
"""---------------------------------------------------------------------------"""
## ------ Tweaking values -------
#neg_correction = 0.1
omega =  1
mu = 1

eps = 1
iters = 0
maxiter = 1000
minimal_error = 1E-10

def vh_to_dens(vh,dens):
    bcs = []
    
    r = SpatialCoordinate(mesh)[0]
    u = TrialFunction(V)
    v = TestFunction(V)        
    a = u*v*dx
    # Works but with oscillations
    #L = (-1.0/(4.0*pi))*(2/r*vh.dx(0)*v-vh.dx(0)*v.dx(0))*dx
    L = (-1.0/(4.0*pi))*(2/r*vh.dx(0)*v-vh.dx(0)*v.dx(0))*dx

    #a = inner(vh,v)*dx
    #L = (-1.0/(4.0*pi))*inner(u.dx(0),v.dx(0))*dx+2/r*inner(u.dx(0),v)*dx
    solve(a == L, dens,bcs)

while eps > minimal_error and iters < maxiter:
    iters += 1 
    
    # Put the initial [density and v_h] in u_k
    (v_hk, u_nk) = split(u_k)
    
    #plotting_sqrt(v_hk, "Hartree potential" )
    plotting_sqrt(u_nk, "In loop PRE solver")   
    
    
    #---- Setting up functionals -------------------
    TF = (5.0/3.0)*CF*u_nk**(2.0/3.0)*pr
    DIRAC = (-4.0/3.0)*CX*pow(u_nk,(1.0/3.0))*pr
    WEIZSACKER = (1.0/8.0*(dot(grad(u_nk),grad(u_nk))/(u_nk**2)*pr+(1.0/4.0*(dot(grad(u_nk),grad(pr)))/u_nk)))
    funcpots = 0
    funcpots = TF \
		+ WEIZSACKER \
        + DIRAC 
      
# =============================================================================
#     # correct for possible negative mu
#     u_i = project(Constant(mu) -Ex - u_nk,V)
#     minval = u_i.vector().min()
#     if minval < 0.0:
#         mu-=minval-1e-14
#     u_i = project(Constant(mu) -Ex - u_nk,V)
#     print("MINIMUM VALUE",u_i.vector().min())
# =============================================================================
        
    
    #---- Solving v_h and u_n ----------------------
    # rotational transformation of nabla^2 v_h = -4 pi n(r)
    F = - v_hk.dx(0)*qr.dx(0)*r/2*dx    \
    + 4*math.pi*u_nk*qr*r/2*dx          \
    + v_hk.dx(0)*qr*dx             
         
    # Second coupled equation: Ts[n] + Exc[n] + Vext(r) - mu = 0
    F = F + funcpots*dx \
    + v_hk*pr*dx        \
    + Ex*pr*dx          \
    - Constant(mu)*pr*dx

    #Calculate Jacobian
    J = derivative(F, u_k, du_trial)
    
    #Assemble system
    A, b = assemble_system(J, -F, bcs_du)
    
    solve(A, du.vector(), b)
    
    
    # Conserve memory by reusing vectors for v_h and u_n to keep dv_h and du_n
    dv_h = v_h                  #step to transfer type info to dv_h
    assign(dv_h, du.sub(0))     #step to transfer values to dv_h
    v_h = None                  #empty v_h
    
    du_n = u_n                  #step to transfer type info to du_n
    assign(du_n, du.sub(1))     #step to transfer values to du_n
    u_n = None                  #empty u_n
    
# =============================================================================
#     dnvec = du_n.vector()
#     dnvec[dnvec<0.0] = 0.0
# =============================================================================
#    plotting_sqrt(du_n,"du - post solver)") #For verifying with TFDW

   
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
    
    
    
# =============================================================================
# #    vh_to_dens(v_h,u_n)
# #---- Ad hoc negative density fix -------
# #    rvec = mesh.coordinates()
# #    nvec = np.array([u_n(v) for v in rvec])
#     nvec = u_n.vector()
#     minval = nvec.min()
#     neg_correction = 0.001
#     
#     print("minval PRE neg fix:",minval)
# 
#     if minval < 0.0:
#         print("Going to add:", -1*minval+neg_correction)
#         #nvec[:]=nvec[:]-minval+neg_correction
#         u_n.vector()[:]= u_n.vector()[:]-minval+neg_correction
#     
#     nvec = u_n.vector()
#     minval = nvec.min()
#     print("minval POST neg fix",minval)
# =============================================================================
    
    nvec = u_n.vector()
    minval = nvec.min()
    print("minval PRE neg fix:",minval)
    #print("u_n vector PRE neg fix: ", nvec.get_local())
    plotting_sqrt(u_n, "PRE NEG FIX")
    
    #[nvec[i] == 0.0 for i in range(len(nvec.get_local()))]    
    
    nvec.get_local()[nvec.get_local()<0.0]=0.0
    print(nvec<0.0)
    print(nvec.get_local())
    #not nvec[nvec<0.0]= 0.0001 ; plotting sqrt(0.0001) != 0 => 0.1    
    nvec = u_n.vector()
    minval = nvec.min()
    print("minval POST neg fix",minval)
    #print("u_n vector POST neg fix: 69308830276194", nvec.get_local())
    plotting_sqrt(u_n, "POST NEG FIX")
     
        
    intn = float(assemble((u_n)*dx(mesh)))
    print("Number of electrons before correction:",intn)
    u_n.vector()[:] = u_n.vector()[:]*N/intn    
    intn = float(assemble((u_n)*dx(mesh)))            
    print("Number of electrons after correction:",intn)
    
    assign(u_k.sub(1),u_n)
    print('Check end of loop, iteration: ', iters)

