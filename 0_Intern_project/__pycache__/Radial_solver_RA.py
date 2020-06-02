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
    y = [u(v)**(2/3)*v*(3*math.pi**2/(2*np.sqrt(2)))**(2/3)  for v in rplot]
    
    #x = numpy.logspace(-5,2,100)
    y = [u(v)**(2/3)*v*(3*math.pi**2/(2*np.sqrt(2)))**(2/3)  for v in rplot]
    pylab.plot(x,y,'rx-')
    pylab.title(title)
    pylab.pause(0.1)
    pylab.xlabel("Alpha * R")
    pylab.ylabel("Psi")
    if wait:
        pylab.waitforbuttonpress()
    
    return     


def plotting_psi_vh(u,title,wait=False):
    a_0 = 1 # Hatree units
    Alpha_ = (4/a_0)*(2*Z/(9*pi**2)**(1/3))
    
    #pylab.clf()
    rplot = mesh.coordinates()
    x = rplot*Alpha_
    
    #x = numpy.logspace(-5,2,100)
    y = [1.0-u(v)*v/Z+0.0 for v in rplot]
    pylab.plot(x,y,'go-')
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
    pylab.waitforbuttonpress()
    pylab.grid
    pylab.xlabel("r")
    pylab.ylabel("n[r]")

    return 

def plotting_normal(u,title):
    pylab.clf()
    
    rplot = mesh.coordinates()
    x = rplot
    #x = numpy.logspace(-5,2,100)
    y = [u(v) for v in rplot]

    #pylab.semilogx(x,y,'bx-')
    pylab.plot(x,y,'kx-')
    pylab.title(title)
    pylab.pause(0.1)
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
n_i = Expression('exp(1.0-8.5*x[0]/radius)', degree=2, radius=rs[-1])
#n_i = Constant(1)

u_n = interpolate(n_i, V)

##plotting_sqrt(u_n, "PRE first solve")

#u = TrialFunction(V)
#v = TestFunction(V)
#a = u.dx(0)*v.dx(0)*dx 
#L = u_n**(3/2)*v*dx
#L = sqrt(u_n)*3*v*dx
#L = u_n*v*dx

#A,  b = assemble_system(a, L, bcs)

#solve(A, u_n.vector(), b)

##plotting_sqrt(u_n, "POST first solve")

#------------Checking amount of electrons ----------------------

#intn = float(assemble((u_n)*dx(mesh)))
#print("[Initial density] Number of electrons before adjustment:", intn)
#u_n.vector()[:] = u_n.vector()[:]*N/intn  
#intn = float(assemble((u_n)*dx(mesh)))
#print("[Initial density] Number of electrons at start after adjustment:",intn)


#nvec = u_n.vector()
#minval = nvec.min()
#print("minval PRE neg fix:",minval)
#print(" PRE nvec values", nvec.get_local())
#plotting_sqrt(u_n,"PRE neg fix ")
# =============================================================================
# #nvec[nvec<0.0]=0.0  #puts index -1 to 0 ?
# print("nvec values", nvec.get_local())
# 
# nvec[nvec.get_local()[nvec<20]] = 0.0
# nvec[nvec.get_local()[-2]] = 0.0
# nvec[nvec.get_local()[-3]] = 0.0
# nvec[nvec.get_local()[-4]] = 0.0
# =============================================================================


#nvec[nvec<0.0] = 0.0   #puts index -1 to 0 ?
#nvec[rs[0]] = 0.0


#nvec = u_n.vector()
#minval = nvec.min()
#print("minval POST neg fix",minval)
#print("POST nvec values", nvec.get_local())
#plotting_log(u_n,"POST neg fix ")
#######----- Initializing boundary conditions on Hartree potential ---###

v_h = interpolate(Constant(0.0), V)

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
startomega = 0.8
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
while eps > minimal_error and iters < maxiter:
    iters += 1 
    
    # Put the initial [density and v_h] in u_k
    (v_hk, u_nk) = split(u_k)
    
    plotting_normal(v_hk, "Hartree potential" )
    plotting_log(u_nk, "In loop PRE solver")       
    
    #---- Setting up functionals -------------------
    TF = (5.0/3.0)*CF*pow(u_nk**2,1.0/3.0)*pr
    DIRAC = (-4.0/3.0)*CX*pow(u_nk,(1.0/3.0))*pr
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
    + 4*math.pi*u_nk*qr*r*dx          \
    + 2*v_hk.dx(0)*qr*dx - v_hk.dx(0)*qr*r*ds(1) + v_hk.dx(0)*qr*r*ds(2)
         
    # Second coupled equation: Ts[n] + Exc[n] + Vext(r) - mu = 0
    F = F + funcpots*dx \
    + v_hk*pr*dx        \
    + Ex*pr*dx          \
    - Constant(mu)*pr*dx

    #Calculate Jacobian
    J = derivative(F, u_k, du_trial)
    
    #Assemble system
    bc_R_du = DirichletBC(W.sub(0), Constant(0.), boundary_R)
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
    
# =============================================================================
#     dnvec = du_n.vector()
#     dnvec[dnvec<0.0] = 0.0
# =============================================================================
    plotting_normal(du_n,"du - post solver)") #For verifying with TFDW

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
    vhvec = v_h.vector()
    minval = nvec.min()
    print("minval PRE neg fix:",minval)    

    elecint = conditional(gt(u_n,0.0),u_n * r * r,0.0)
    intn1 = 4.0*pi*float(assemble((elecint)*dx(mesh)))
    print("Electron count max:",intn1)

    elecint = conditional(lt(r,radius),u_n * r * r,0.0)
    intn2 = 4.0*pi*float(assemble((elecint)*dx(mesh)))
    print("Electron count min:",intn2)

    #elecint = conditional(gt(u_n,0.0),u_n *r*r , 1E-8)
    # if u_n > 0.0 elecint == u_n*r^2 
    # else          elecint == 0.0
    #intn3 =4*math.pi*float(assemble((elecint)*ds))
    #print("Electron count min2:",intn3)
    
    if intn1 <= 1e-4:
        print("Electron count too small")
        u_n = interpolate(Constant(1.0), V)

    plotting_log(u_n, "PRE NEG FIX")

    #x = rs_outer
    #y = [u_n(rv) for rv in x]
    #radius = x[-2]
    #radius2 = x[-1]
    #radval = 1e-12
    #for i in range(len(y)):
    #    if y[i] <= 1e-10:
    #        if i == 0:
    #            radius = 0.0
    #            pass
    #        else:
    #            radius = x[i]*3.0/4.0
    #            radius2 = x[i]*4.0/5.0
    #            radval = u_n(radius)
    #            break

    #print("RADIUS:",radius)

    #if radius == 0.0:        
    #    assign(u_n,interpolate(Constant(1), V))
    #elif radius < x[-1]:
        #fitexpr = smoothstep(radius,radius+1.0,r)*radval + (1.0-smoothstep(radius,radius+1.0,r))*conditional(gt(u_n,radval),u_n,radval)
        #conditional(gt(r,radius),1e-10,u_n)
        #fitfunc = project(fitexpr, V)
        #assign(u_n,fitfunc)

    #x = [rv for rv in rs if rv < radius]
    #y = [u_n(rv) for rv in rs if rv < radius]
        
    #params = np.polyfit(x, np.log(y), 1)
    #fitexpr = Expression('exp(p1)*exp(p0*x[0])', degree=2, p1=params[1],p0=params[0])
    #fitexpr2 = smoothstep(radius,radius2,r)*fitexpr + (1.0-smoothstep(radius,radius2,r))*u_n
#   # fitexpr3 = conditional(gt(fitexpr2,1e-14),fitexpr2,1e-14)
    #fitfunc = project(fitexpr2, V)
    #assign(u_n,fitfunc)

    #for i in range(len(nvec.get_local())):
    #    if nvec[i] < 1e-8:
    #        nvec[i] = 1e-8
    #        
    #plotting_log_keep(u_n, "DENS POST NEG FIX")

    #nvec = u_n.vector()
    #minval = nvec.min()
    #print("minval POST neg fix:",minval)    
    
    #print("u_n vector PRE neg fix: ", nvec.get_local())

    #fitexpr = conditional(gt(u_n,1e-10),u_n,1e-10)
    #fitfunc = project(fitexpr, V)
    #assign(u_n,fitfunc)


    #[nvec[i] == 0.0 for i in range(len(nvec.get_local()))]    
    
    #nvec.get_local()[nvec.get_local()<0.0]=0.0
    #print(nvec<0.0)
    #print(nvec.get_local())
    #not nvec[nvec<0.0]= 0.0001 ; plotting sqrt(0.0001) != 0 => 0.1    
    #nvec = u_n.vector()
    #minval = nvec.min()
    #print("minval POST neg fix",minval)
    #print("u_n vector POST neg fix: 69308830276194", nvec.get_local())
    #plotting_sqrt(u_n, "POST NEG FIX")

    plotting_psi(u_n,"PSI",wait=False)    
    plotting_psi_vh(v_h,"PSI", wait=True)    
    
    #intn = float(assemble((u_n)*dx(mesh)))
    #print("Number of electrons before correction:",intn)
    #u_n.vector()[:] = u_n.vector()[:]*N/intn    
    #intn = float(assemble((u_n)*dx(mesh)))            
    #print("Number of electrons after correction:",intn)

    #nvec[:] = nvec[:]*N/intn1    
    #vhvec[:] = vhvec[:]*N/intn1
    
    assign(u_k.sub(1),u_n)

    #offset = v_h(rs[-1])
    #new_v_h = v_h - offset
    #new_v_h_proj = project(new_v_h, V)
    #assign(v_h,new_v_h_proj)
    
    assign(u_k.sub(0),v_h)
    
    print('Check end of loop, iteration: ', iters)

