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
Z = 1   # Hydrogen
N = Z 		  # Neutral

#Element Ne
#Z = Constant(10) # Neon
#N = Z 		  # Neutral

#Element Kr
#Z = Constant(36) # Krypton
#N = Z 		  # Neutral 

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

def smoothzero(x, delta):
    x[x < 10.0/delta] = 1.0/delta*np.log(1+np.exp(x[x<10.0/delta]*delta))

def smoothstep(edge0, edge1, x):
  x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0) 
  return x * x * (3 - 2 * x)

def clamp(x, lowerlimit,  upperlimit):
    return conditional(lt(x,lowerlimit),lowerlimit,conditional(gt(x,upperlimit),upperlimit,x))
    

def plotting_psi(u,title,wait=False):
    a_0 = 1 # Hatree units
    Alpha_ = (4/a_0)*(2*Z/(9*pi**2)**(1/3))
    
    pylab.clf()
    rplot = mesh.coordinates()
    x = rplot*Alpha_
    try:
        y = [u(v)**(2/3)*v*(3*math.pi**2/(2*np.sqrt(2)))**(2/3)  for v in rplot]
    except:
        y = [0.0 for v in rplot]    
    #x = numpy.logspace(-5,2,100)
    #y = [u(v)**(2/3)*v*(3*math.pi**2/(2*np.sqrt(2)))**(2/3)  for v in rplot]
    pylab.plot(x,y,'kx-')
    pylab.title(title)
    pylab.pause(0.1)
    pylab.xlabel("Alpha * R")
    pylab.ylabel("Psi")
    if wait:
        pylab.waitforbuttonpress()
    
    return     

def plotting_log_keep(u,title):
    rplot = mesh.coordinates()
    x = rplot
    #x = numpy.logspace(-5,2,100)
    y = [u(v) for v in rplot]

    pylab.semilogy(x,y,'rx-')
    #pylab.plot(x,y,'kx-')
    pylab.title(title)
    pylab.pause(0.1)
    #pylab.waitforbuttonpress()
    pylab.grid
    pylab.xlabel("r")
    pylab.ylabel("n[r]")

    return 

def plotting_normal(u,title,wait=False):
    pylab.clf()
    
    rplot = mesh.coordinates()
    x = rplot
    #x = numpy.logspace(-5,2,100)
    try:
        y = [u(v) for v in rplot]
    except:
        y = [0.0 for v in rplot]
    
    #pylab.semilogx(x,y,'bx-')
    pylab.plot(x,y,'kx-')
    pylab.plot(rplot[-1],u(rplot[-1]),'bo')
    pylab.title(title)
    pylab.pause(0.1)
    if wait:
        pylab.waitforbuttonpress()
    pylab.grid
    pylab.xlabel("r")
    pylab.ylabel("n[r]")

    return 


def plotting_log(u,title):
    pylab.clf()
    
    rplot = mesh.coordinates()
    x = rplot
    #x = numpy.logspace(-5,2,100)
    y = [u(v) for v in rplot]

    pylab.semilogy(x,y,'bx-')
    #pylab.plot(x,y,'kx-')
    pylab.title(title)
    pylab.pause(0.1)
    #pylab.waitforbuttonpress()
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
    #pylab.waitforbuttonpress()
    pylab.pause(0.1)
    pylab.xlabel("SQRT(R)")
    pylab.ylabel("R * SQRT(density")
    
    return 
     
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating the mesh + Function Spaces + defining type of element
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------""" 


#rs = np.arange(0.1,25.0,0.01)
rs = np.arange(0.01,3.0/Alpha,0.01)
radius = rs[-1]
r_inner = 0.0
rs_outer = [x for x in rs if x > r_inner]

start_x = rs[0]
end_x = rs[-1]
amount_vertices = len(rs)
mesh = IntervalMesh(amount_vertices,start_x, end_x) # Splits up the interval [0,1] in (n) elements 

#print("MES COORDS",np.shape(mesh.coordinates()))

#Creation of Function Space
P1 = FiniteElement('P', mesh.ufl_cell(), 2)
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
#n_i = Expression('a*exp(-b*pow(x[0], 2))', degree = 2, a = 1, b=0.05)
#n_i = Expression('rho0*exp(-fac*Z*x[0])', degree=2, Z=Z,fac=1.8,rho0=5832)
#n_i = Constant(-0.0)
n_i = Expression('1.0-5*x[0]/radius', degree=2, radius=rs[-1])

u_n = interpolate(n_i, V)
v_h = interpolate(Constant(0), V)

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

#bc_L_du = DirichletBC(V, Constant(0), boundary_L)
#bc_L_du = None
bc_R_du = DirichletBC(V, Constant(0), boundary_R)
#bcs_du = [bc_L_du, bc_R_du]
bcs_du = [bc_R_du]
"""---------------------------------------------------------------------------"""
## ------ Tweaking values -------
#neg_correction = 0.1
startomega = 1.0
mu = 0

eps = 1
iters = 0
maxiter = 1000
minimal_error = 1E-9



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

omega = startomega
wait = True
while eps > minimal_error and iters < maxiter:
    iters += 1 
    
    # Put the initial [density and v_h] in u_k
    (v_hk, u_nk) = split(u_k)
    
    plotting_normal(v_hk, "Hartree potential", wait=False )
    #plotting_log(u_nk, "In loop PRE solver")       
    plotting_normal(u_n, "LOG DENSITY PRE SOLVER", wait=False )
    
    #---- Setting up functionals -------------------
    TF = (5.0/3.0)*CF*exp(2*u_nk/3)*pr
    DIRAC = (-4.0/3.0)*CX*exp(u_nk/3.0)*pr
    #WEIZSACKER = (1.0/8.0*(dot(grad(u_nk),grad(u_nk))/(u_nk**2)*pr+(1.0/4.0*(dot(grad(u_nk),grad(pr)))/u_nk)))
    WEIZSACKER = (1.0/8.0*(u_nk.dx(0))*u_nk.dx(0)/(u_nk**2)*pr+(1.0/4.0*(u_nk.dx(0))*pr.dx(0))/u_nk)
    #funcpots = TF + WEIZSACKER + DIRAC
    funcpots = TF
      
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
    F = - v_hk.dx(0)*qr.dx(0)*r*dx    \
    + 4*math.pi*exp(u_nk)*qr*r*dx     \
    + 2*v_hk.dx(0)*qr*dx # + v_hk.dx(0)*qr*ds(0)
         
    # Second coupled equation: Ts[n] + Exc[n] + Vext(r) - mu = 0
    F = F + funcpots*dx \
    + v_hk*pr*dx        \
    + Ex*pr*dx          \
    - Constant(mu)*pr*dx

    #Calculate Jacobian
    J = derivative(F, u_k, du_trial)
    
    #Assemble system
    bc_R_du = DirichletBC(W.sub(0), Constant(0), boundary_R)
    bcs_du = [bc_R_du]
    #bcs_du = []

    A, b = assemble_system(J, -F, bcs=bcs_du)
    
    solve(A, du.vector(), b)
    
    
    # Conserve memory by reusing vectors for v_h and u_n to keep dv_h and du_n
    dv_h = v_h                  #step to transfer type info to dv_h
    assign(dv_h, du.sub(0))     #step to transfer values to dv_h
    v_h = None                  #empty v_h
    
    du_n = u_n                  #step to transfer type info to du_n
    assign(du_n, du.sub(1))     #step to transfer values to du_n
    u_n = None                  #empty u_n
    
    plotting_normal(du_n,"du - post solver)",wait=wait) #For verifying with TFDW

    #---- Calculate the Error -----------------------
    epsexpr = du_n**2
    eps = float(assemble((epsexpr)*dx(mesh)))    
    if math.isnan(eps):
        raise Exception("Residual error is NaN")
    print("EPS is:",eps)
    # Once we actually converge, we should no longer dampen the newton iterations
    if eps < 0.1:
        omega = 1.0
    #else:
    #    omega = min(1.0,abs(10.0/eps))
    if eps < 1e-4:
        wait = True
    
    
    #--- Assigning the last calculated density to nlast
    assign(nlast,u_k.sub(1))
        
    #---- Taking the step for u_n and v_h ------------
    #print("U_k before",u_k(1.0))
    u_k.vector()[:] = u_k.vector()[:] + omega*du.vector()[:]
    #print("U_k after",u_k(1.0))
    
    # Conserve memory by reusing vectors for u_n, v_h also as du_n, dv_h
    v_h = dv_h 
    assign(v_h, u_k.sub(0))
    dv_h = None
    
    u_n = du_n 
    assign(u_n, u_k.sub(1))
    du_n = None

    nvec = u_n.vector()
    vhvec = v_h.vector()

    elecint = exp(u_n)*r*r
    intn1 = 4.0*pi*float(assemble((elecint)*dx(mesh)))
    print("Electron count:",intn1)
    
    #if intn1 <= 1e-4:
    #    print("Electron count too small")
    #    u_n = interpolate(Constant(1.0), V)

    plotting_normal(u_n, "LOG DENSITY", wait=wait)
    plotting_psi(exp(u_n),"PSI", wait=wait)    

    print("v_h end value:",v_h(rs[-1]))
    
#    if eps < 0.1:
#        fixfactor = N/intn1
#    #    new_v_h = v_h*fixfactor
#    #    offset = v_h(rs[-1])
#        offset = (1.0-fixfactor)*0.1
#        new_v_h = v_h + offset
#        new_v_h_proj = project(new_v_h, V)
#        assign(v_h,new_v_h_proj)
#        assign(u_k.sub(0),v_h)        
#        print("v_h after fix",v_h(rs[-1]))
#
#        new_u_n = u_n + ln(fixfactor)
#        new_u_n_proj = project(new_u_n, V)
#        assign(u_n,new_u_n_proj)
#        assign(u_k.sub(0),v_h)                
#
#        elecint = exp(u_n)*r*r
#        intn1 = 4.0*pi*float(assemble((elecint)*dx(mesh)))
#        print("Electron count after fix:",intn1)
        
    #adjexpr = conditional(gt(r,rs[-3]),-30,u_n)
    #adjfunc = project(adjexpr, V)
    #assign(u_n,adjfunc)
    
    #assign(u_k.sub(1),u_n)

    
    print('Check end of loop, iteration: ', iters)

