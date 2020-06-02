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
Alpha = (4/a_0)*(2*Z/(9*pi**2))**(1/3)


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
    pylab.clf()
    rplot = mesh.coordinates()
    x = rplot*Alpha
    y = [u(v)**(2/3)*v*(3*math.pi**2/(2*np.sqrt(2)))**(2/3)  for v in rplot]
    pylab.plot(x,y,'rx-')
    pylab.title(title)
    pylab.grid()
    pylab.pause(0.1)
    pylab.xlabel(r"$\alpha r$", fontsize=18)
    pylab.ylabel(r"$\Psi$", fontsize=18)
    if wait:
        pylab.waitforbuttonpress()
    
    return     


def plotting_psi_keep(u,title,wait=False):
    #pylab.clf()
    rplot = mesh.coordinates()
    x = rplot*Alpha
    
    y = [1.0-u(v)*v/Z+0.0 for v in rplot]
    pylab.plot(x,y,'go-')
    pylab.title(title)
    pylab.grid()
    pylab.pause(0.1)
    pylab.xlabel(r"$\alpha r$", fontsize=18)
    pylab.ylabel(r"$\Psi$", fontsize=18)
    if wait:
        pylab.waitforbuttonpress()
    
    return     


def plotting_log_keep(u,title, wait= False):
    rplot = mesh.coordinates()
    x = rplot
    y = [u(v) for v in rplot]

    pylab.semilogy(x,y,'rx-')

    pylab.title(title)
    pylab.pause(0.1)
    if wait:
        pylab.waitforbuttonpress()
    pylab.grid()
    pylab.xlabel("r", fontsize=18)
    pylab.ylabel("n[r]", fontsize=18)

    return 

def plotting_normal(u,title, wait= False):
    pylab.clf()
    
    rplot = mesh.coordinates()
    x = rplot
    y = [u(v) for v in rplot]


    pylab.plot(x,y,'kx-')
    pylab.title(title)
    pylab.pause(0.1)
    if wait:
        pylab.waitforbuttonpress()
    pylab.grid()
    pylab.xlabel("r", fontsize=18)
    pylab.ylabel("n[r]", fontsize=18)

    return 

def plotting_log(u,title, wait= False):
    pylab.clf()
    
    rplot = mesh.coordinates()
    x = rplot
    y = [u(v) for v in rplot]

    pylab.semilogy(x,y,'bx-')
    #pylab.plot(x,y,'kx-')
    pylab.title(title)
    pylab.pause(0.1)
    if wait:
        pylab.waitforbuttonpress()
    pylab.grid()
    pylab.xlabel("r", fontsize=18)
    pylab.ylabel("n[r]", fontsize=18)

    return 

def plotting_sqrt(u,title, wait= False):
    pylab.clf()
    rplot = mesh.coordinates()
    x = np.sqrt(rplot)
    y = [v*sqrt(u(v)) for v in rplot] 
    
    pylab.plot(x,y,'kx-')
    pylab.title(title)
    if wait:
        pylab.waitforbuttonpress()
    pylab.pause(0.1)
    pylab.grid()
    pylab.xlabel(r"$\sqrt{R}$", fontsize=18)
    pylab.ylabel(r"$R \sqrt{n(r)}$", fontsize=18)
    
    return 
     
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Creating the mesh + Function Spaces + defining type of element
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------""" 


rs = np.arange(1e-5, 25.0, 0.01)
#rs = np.arange(0.01, 3.0/Alpha,0.01)
radius = rs[-1]
r_inner = 0.0
rs_outer = [x for x in rs if x > r_inner]

start_x = rs[0]
end_x = rs[-1]
amount_vertices = len(rs)
mesh = IntervalMesh(amount_vertices,start_x, end_x) 

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

class LeftBoundary(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and near(x[0], start_x, tol)

class RightBoundary(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and near(x[0], end_x, tol)    

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
left_boundary = LeftBoundary()
left_boundary.mark(boundaries,1)
right_boundary = RightBoundary()
right_boundary.mark(boundaries,2)
ds = Measure("ds", subdomain_data=boundaries)

    
"""-------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------- 
                    Defning external potential v[r] 
                    Initial density n_1[r]
                    Hartree potential v_h
----------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------"""
# External potential
Ex = -Z/r

## ---  Initial density 
n_i = Expression('exp(1.0-8.5*x[0]/radius)', degree=2, radius=rs[-1])
#n_i = Constant(1)
u_n = interpolate(n_i, V)

## --- Initial Hartree potential 
v_h = interpolate(Expression('Z/x[0]',Z=Z,degree=2), V)


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

nlast = Function(V)

"""---------------------------------------------------------------------------"""
## ------ Tweaking values -------
#neg_correction = 0.1
startomega = 0.8
mu = 0

eps = 1
iters = 0
maxiter = 1000
minimal_error = 1E-9



omega = startomega
while eps > minimal_error and iters < maxiter:
    iters += 1 
    
    # Put the initial [density and v_h] in u_k
    (v_hk, u_nk) = split(u_k)
    
#    plotting_normal(v_hk, "Hartree potential" )
#    plotting_log(u_nk, "Density LOG - PRE solver")       
#    plotting_sqrt(u_nk, 'Density TFDW PRE solver')
    
    #---- Setting up functionals -------------------
    TF = (5.0/3.0)*CF*pow(u_nk**2,1.0/3.0)*pr
    DIRAC = (-4.0/3.0)*CX*pow(u_nk,(1.0/3.0))*pr
    #WEIZSACKER = (1.0/8.0*(dot(grad(u_nk),grad(u_nk))/(u_nk**2)*pr+(1.0/4.0*(dot(grad(u_nk),grad(pr)))/u_nk)))
    WEIZSACKER = (1.0/8.0*(u_nk.dx(0))*u_nk.dx(0)/(u_nk**2)*pr+(1.0/4.0*(u_nk.dx(0))*pr.dx(0))/u_nk)
    
    funcpots = TF + WEIZSACKER + DIRAC
    #funcpots = TF + DIRAC
    #funcpots = TF + WEIZSACKER
    #funcpots = TF 
     
    
    #---- Solving v_h and u_n ----------------------
    # rotational transformation of nabla^2 v_h = -4 pi n(r)
    F = - r*v_hk.dx(0)*qr.dx(0)*dx    \
    + 4*math.pi*u_nk*r*qr*dx          \
    + (2)*v_hk.dx(0)*qr*dx - r*v_hk.dx(0)*qr*ds(1) + r*v_hk.dx(0)*qr*ds(2)
         
    # Second coupled equation: Ts[n] + Exc[n] + Vext(r) - mu = 0
    F = F + funcpots*dx \
    + v_hk*pr*dx        \
    + Ex*pr*dx          \
    - Constant(mu)*pr*dx

    #Calculate Jacobian
    J = derivative(F, u_k, du_trial)
    
    #Assemble system
    bc_R_du = DirichletBC(W.sub(1), (0), boundary_R)
   #bc_R_du = DirichletBC(W.sub(0), (Z/r), boundary_R)
    bcs_du = [bc_R_du]
    A, b = assemble_system(J, -F, bcs_du)
    
    solve(A, du.vector(), b)
    
    
    
    
    # Conserve memory by reusing vectors for v_h and u_n to keep dv_h and du_n
    dv_h = v_h                  #step to transfer type info to dv_h
    assign(dv_h, du.sub(0))     #step to transfer values to dv_h
    v_h = None                  #empty v_h
    
    du_n = u_n                  #step to transfer type info to du_n
    assign(du_n, du.sub(1))     #step to transfer values to du_n
    u_n = None                  #empty u_n
    
    
    

    #---- Calculate the Error -----------------------
    epsexpr = conditional(lt(r,radius),du_n**2,0.0)
    eps = float(assemble((epsexpr)*dx(mesh)))    
    if math.isnan(eps):
        raise Exception("Residual error is NaN")
    print("EPS is:",eps)
    # Once we actually converge, we should no longer dampen the newton iterations
    if eps < 0.1:
        omega = 1.0
    
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
    

    nvec = u_n.vector()
    vhvec = v_h.vector()
    minval = nvec.min()
    print("minval PRE neg fix:",minval)    
#    plotting_log(u_n, "Density PRE correction", wait=False)
   
    tail = False
    if tail == True:
        
        x = rs_outer
        y = [u_n(rv) for rv in x]
        radius = x[-1]
        radval = 1e-12   
        for i in range(len(y)):
            if y[i] <= 1e-10:
                if i == 0:
                    radius = 0.0
                    pass
                else:
                    radius = x[i]*3.0/4.0
                    radval = u_n(radius)
                    break

        print("RADIUS:",radius)

        if radius == 0.0:        
            assign(u_n,interpolate(Constant(1), V))
            
        elif radius < x[-1]:
            fitexpr = smoothstep(radius,radius+1.0,r)*radval + (1.0-smoothstep(radius,radius+1.0,r))*conditional(gt(u_n,radval),u_n,radval)
            conditional(gt(r,radius),1e-10,u_n)
            fitfunc = project(fitexpr, V)
            assign(u_n,fitfunc)
    
#    plotting_log_keep(u_n, "Density POST correction",wait=False)
    
    elecint = conditional(gt(u_n,0.0),u_n * r * r,0.0)
    intn1 = 4.0*pi*float(assemble((elecint)*dx(mesh)))
    print("Electron count max:",intn1)

    elecint = conditional(lt(r,radius),u_n * r * r,0.0)
    intn2 = 4.0*pi*float(assemble((elecint)*dx(mesh)))
    print("Electron count min:",intn2)

    
    if intn1 <= 1e-4:
        print("Electron count too small")
        u_n = interpolate(Constant(1.0), V)

#    plotting_psi(u_n,"Density PSI",wait=False)    
#    plotting_psi_keep(v_h,"Hartree potential PSI", wait=False)    

    plotting_sqrt(u_n, "Density Post solver", wait= False)
#    plotting_sqrt(v_h, "Hartree potential Post solver", wait= False)

#    plotting_normal(u_n, "Density Post solver", wait=False)
#    plotting_normal(v_h, " Hartree potential Post solver", wait=False)
        

#    plotting_log_keep(u_n, "Density POST correction",wait=True)
#    plotting_log(u_n, "Density Post solver", wait=True)
#    plotting_log_keep(v_h, "Hartree Potential post solver", wait=True)    
    
    assign(u_k.sub(1),u_n) 
    assign(u_k.sub(0),v_h)
    
    print('Check end of loop, iteration: ', iters)








#plotting_sqrt(nlast, " Final density") 
#plotting_psi(nlast, " Final density PSI")

h_to_ev = 27.21138386

#Born - Oppenheimer approximation.
ionion_energy = 0.0

ionelec_energy = 4*math.pi*float(assemble(nlast*r*r*dx))

#---- Calculate electron-electron interaction energy
          
elecelec_energy = 0.5*float(assemble(nlast*r*r*dx))
        
#---- Calculate functional energy

func_energy_expression = 1.0/8.0*nlast.dx(0)/nlast\
                         + CF*pow(nlast,(5.0/3.0))\
                         - CX*pow(nlast,(4.0/3.0))

functional_energy = float(assemble(func_energy_expression*dx))
#print('check type of the functional energy', type(functional_energy))
#print('check type of the ionion energy', type(ionion_energy))
#print('check type of the ionelec energy', type(ionelec_energy))
#print('check type of the elecelec energy', type(elecelec_energy))
total = ionion_energy + ionelec_energy + elecelec_energy + functional_energy

#--- printing energies
print ("==== Resulting energies (hartree to ev): ================")
print ("Ion-ion:        % 10.4f"%(ionion_energy*h_to_ev))
print ("Ion-elec:       % 10.4f"%(ionelec_energy*h_to_ev))
print ("Elec-elec (Eh): % 10.4f"%(elecelec_energy*h_to_ev))
print ("Functional:     % 10.4f"%(functional_energy*h_to_ev))
print ("==============================================")
print ("Total energy WITH tail:   % 10.4f"%(total*27.21138386))
print ("Total (Born-Oppenheimer approx):  % 10.4f"%((ionelec_energy + elecelec_energy + functional_energy)*27.21138386))
print ("==============================================")

#Error bar on energies
field2 = conditional(lt(r,radius),nlast,0.0)

ionelec_energy = 4*math.pi*float(assemble(field2*r*r*dx))

#---- Calculate electron-electron interaction energy
          
elecelec_energy = 0.5*float(assemble(field2*r*r*dx))
        
#---- Calculate functional energy

func_energy_expression = 1.0/8.0*nlast.dx(0)/nlast\
                         + CF*pow(nlast,(5.0/3.0))\
                         - CX*pow(nlast,(4.0/3.0))

functional_energy = float(assemble(func_energy_expression*dx))
#print('check type of the functional energy', type(functional_energy))
#print('check type of the ionion energy', type(ionion_energy))
#print('check type of the ionelec energy', type(ionelec_energy))
#print('check type of the elecelec energy', type(elecelec_energy))
total = ionion_energy + ionelec_energy + elecelec_energy + functional_energy


print ("==============================================")
print ("Total energy NO tail:   % 10.4f"%(total*27.21138386))
print ("Total (Born-Oppenheimer approx):  % 10.4f"%((ionelec_energy + elecelec_energy + functional_energy)*27.21138386))
print ("==============================================")


